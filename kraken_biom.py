#!/usr/bin/env python
# coding: utf-8
"""
Create BIOM-format tables (http://biom-format.org) from 
Kraken output (http://ccb.jhu.edu/software/kraken/).
"""
from __future__ import absolute_import, division, print_function
from typing import Dict, List, OrderedDict, Any, Tuple
from pathlib import Path
import argparse
from collections import OrderedDict
import csv
from datetime import datetime as dt
from gzip import open as gzip_open
import os.path as osp
import re
import sys
from textwrap import dedent as twdd

import numpy as np
import pandas as pd
from biom.table import Table

try:
    import h5py
    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

__author__ = "Shareef M. Dabdoub"
__copyright__ = "Copyright 2016, Shareef M. Dabdoub"
__credits__ = ["Shareef M. Dabdoub", "Akshay Paropkari", 
               "Sukirth Ganesan", "Purnima Kumar"]
__license__ = "MIT"
__url__ = "http://github.com/smdabdoub/kraken-biom"
__maintainer__ = "Shareef M. Dabdoub"
__email__ = "dabdoub.2@osu.edu"
__version__ = '1.2.1'  # Updated version for Python 3.12 compatibility


field_names = ["pct_reads", "clade_reads", "taxon_reads", 
               "rank", "ncbi_tax", "sci_name"]
ranks = ["D", "P", "C", "O", "F", "G", "S", "SS"]


def tax_fmt(tax_lvl: Dict[str, str], end: int) -> List[str]:
    """
    Create a string representation of a taxonomic hierarchy (QIIME format).

    Args:
        tax_lvl: Dictionary keyed on the entries in ranks
        end: The end rank index (0-based indexing)
    
    Returns:
        List of formatted taxonomy strings
    """
    if "S" in tax_lvl:
        if "G" in tax_lvl and tax_lvl["S"].startswith(tax_lvl["G"]):
            tax_lvl["S"] = tax_lvl["S"][len(tax_lvl["G"])+1:]
    if "SS" in tax_lvl:
        if "S" in tax_lvl and tax_lvl["SS"].startswith(tax_lvl["S"]):
            tax_lvl["SS"] = tax_lvl["SS"][len(tax_lvl["S"])+1:]
    
    tax = [f"{r.lower()}__{tax_lvl.get(r, '')}" for r in ranks[:end+1]]
    tax.extend([f"{r.lower()}__" for r in ranks[end+1:-1]])

    if tax[0].startswith('d'):
        tax[0] = "k" + tax[0][1:]

    return tax


def parse_tax_lvl(entry: Dict[str, str], tax_lvl_depth: List[Tuple[str, str]] = None) -> Dict[str, str]:
    """
    Parse a single kraken-report entry and return a dictionary of taxa for its
    named ranks.

    Args:
        entry: Attributes of a single kraken-report row
        tax_lvl_depth: Running record of taxon levels encountered in previous calls
    
    Returns:
        Dictionary mapping rank codes to taxon names
    """
    if tax_lvl_depth is None:
        tax_lvl_depth = []
        
    depth_and_name = re.match(r'^( *)(.*)', entry['sci_name'])
    depth = len(depth_and_name.group(1))//2
    name = depth_and_name.group(2)
    
    del tax_lvl_depth[depth:]
    
    erank = entry['rank']
    if erank == '-' and depth > 8 and tax_lvl_depth and tax_lvl_depth[-1][0] == 'S':
        erank = 'SS'
    tax_lvl_depth.append((erank, name))

    return {x[0]: x[1] for x in tax_lvl_depth if x[0] in ranks}


def parse_kraken_report(kdata: List[Dict[str, str]], max_rank: str, min_rank: str) -> Tuple[OrderedDict, OrderedDict]:
    """
    Parse a single output file from the kraken-report tool.

    Args:
        kdata: Contents of the kraken report file
        max_rank: Maximum taxonomic rank to include
        min_rank: Minimum taxonomic rank to include
    
    Returns:
        Tuple of (counts OrderedDict, taxa OrderedDict)
    """
    taxa: OrderedDict[str, List[str]] = OrderedDict()
    counts: OrderedDict[str, int] = OrderedDict()
    max_rank_idx = ranks.index(max_rank)
    min_rank_idx = ranks.index(min_rank)

    for entry in kdata:
        tax_lvl = parse_tax_lvl(entry)
        erank = entry['rank'].strip()
        
        if 'SS' in tax_lvl:
            erank = 'SS'

        if erank in ranks:
            r = ranks.index(erank)     
            if min_rank_idx >= r >= max_rank_idx:
                taxon_reads = int(entry["taxon_reads"])
                clade_reads = int(entry["clade_reads"])
                if taxon_reads > 0 or (clade_reads > 0 and erank == min_rank):
                    taxa[entry['ncbi_tax']] = tax_fmt(tax_lvl, r)
                    counts[entry['ncbi_tax']] = clade_reads if erank == min_rank else taxon_reads

    return counts, taxa


def process_samples(kraken_reports_fp: List[str], max_rank: str, min_rank: str) -> Tuple[OrderedDict, OrderedDict]:
    """
    Parse all kraken-report data files into sample counts dict
    and store global taxon id -> taxonomy data.
    """
    taxa: OrderedDict[str, List[str]] = OrderedDict()
    sample_counts: OrderedDict[str, Dict[str, int]] = OrderedDict()
    
    for krep_fp in kraken_reports_fp:
        if not osp.isfile(krep_fp):
            raise RuntimeError(f"ERROR: File '{krep_fp}' not found.")

        sample_id = osp.splitext(osp.split(krep_fp)[1])[0]

        try:
            with open(krep_fp, "rt", encoding='utf-8') as kf:
                kdr = csv.DictReader(kf, fieldnames=field_names, delimiter="\t")
                kdata = [entry for entry in kdr][1:]
        except OSError as oe:
            raise RuntimeError(f"ERROR: {oe}")

        scounts, staxa = parse_kraken_report(kdata, max_rank=max_rank, min_rank=min_rank)
        taxa.update(staxa)
        sample_counts[sample_id] = scounts

    return sample_counts, taxa


def process_metadata(sample_counts: Dict[str, Dict[str, int]], metadata: str) -> List[Dict[str, Any]]:
    """
    Read the sample metadata file or create dummy metadata.
    """
    if metadata:
        metadata_frame = pd.read_csv(metadata, sep='\t', dtype=str)
        samples_imported = list(sample_counts.keys())
        metadata_frame.index = metadata_frame[metadata_frame.columns[0]]
        return metadata_frame.loc[samples_imported].to_dict(orient='records')
    
    return [{"Id": key} for key in sample_counts.keys()]


def create_biom_table(sample_counts: Dict[str, Dict[str, int]], 
                     taxa: Dict[str, List[str]], 
                     sample_metadata: List[Dict[str, Any]]) -> Table:
    """
    Create a BIOM table from sample counts and taxonomy metadata.
    """
    data = np.array([[sample_counts[sid].get(taxid, 0) for sid in sample_counts] 
                     for taxid in taxa], dtype=int)
    tax_meta = [{'taxonomy': taxa[taxid]} for taxid in taxa]
    
    gen_str = f"kraken-biom v{__version__} ({__url__})"

    return Table(data, list(taxa), list(sample_counts), tax_meta,
                 type="OTU table", create_date=str(dt.now().isoformat()),
                 generated_by=gen_str, input_is_dense=True,
                 sample_metadata=sample_metadata)


def write_biom(biomT: Table, output_fp: str, fmt: str = "hdf5", gzip: bool = False) -> str:
    """
    Write the BIOM table to a file.
    """
    opener = open
    mode = 'w'
    if gzip and fmt != "hdf5":
        if not output_fp.endswith(".gz"):
            output_fp += ".gz"
        opener = gzip_open
        mode = 'wt'

    if fmt == "hdf5":
        opener = h5py.File

    with opener(output_fp, mode) as biom_f:
        if fmt == "json":
            biomT.to_json(biomT.generated_by, direct_io=biom_f)
        elif fmt == "tsv":
            biom_f.write(biomT.to_tsv())
        else:
            biomT.to_hdf5(biom_f, biomT.generated_by)

    return output_fp


def write_otu_file(otu_ids: List[str], fp: str) -> None:
    """
    Write out OTU IDs to a file.
    """
    fpdir = osp.split(fp)[0]
    if fpdir and not osp.isdir(fpdir):
        raise RuntimeError(f"Specified path does not exist: {fpdir}")

    with open(fp, 'wt', encoding='utf-8') as outf:
        outf.write('\n'.join(otu_ids))


def handle_program_options() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description=twdd(__doc__),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('kraken_reports', nargs='*',
                        help="Results files from the kraken-report tool.")
    parser.add_argument('-k', '--kraken_reports_fp',
                        help="Folder containing kraken reports")
    parser.add_argument('--max', default="O", choices=ranks,
                        help="Maximum taxonomic rank (default: O)")
    parser.add_argument('--min', default="S", choices=ranks,
                        help="Minimum taxonomic rank (default: S)")
    parser.add_argument('-o', '--output_fp', default="table.biom",
                        help="Output BIOM file path (default: table.biom)")
    parser.add_argument('-m', '--metadata',
                        help="Sample metadata file path (TSV format)")
    parser.add_argument('--otu_fp',
                        help="Output file path for OTU IDs")
    parser.add_argument('--fmt', default="hdf5", 
                        choices=["hdf5", "json", "tsv"],
                        help="Output format (default: hdf5)")
    parser.add_argument('--gzip', action='store_true',
                        help="Compress output (except HDF5)")
    parser.add_argument('--version', action='version',
                        version=f"kraken-biom version {__version__}, {__url__}")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Print status messages")

    return parser.parse_args()


def main() -> None:
    """Main program function."""
    args = handle_program_options()

    if args.fmt == 'hdf5' and not HAVE_H5PY:
        args.fmt = 'json'
        print(twdd("Library 'h5py' not found. Defaulting to BIOM 1.0 (JSON)."))

    if ranks.index(args.max) > ranks.index(args.min):
        sys.exit(f"ERROR: Max and Min ranks are out of order: {args.max} < {args.min}")

    reports = args.kraken_reports or []
    if args.kraken_reports_fp:
        reports.extend(str(p) for p in Path(args.kraken_reports_fp).glob('*'))

    sample_counts, taxa = process_samples(reports, max_rank=args.max, min_rank=args.min)
    sample_metadata = process_metadata(sample_counts, args.metadata)
    biomT = create_biom_table(sample_counts, taxa, sample_metadata)
    out_fp = write_biom(biomT, args.output_fp, args.fmt, args.gzip)

    if args.otu_fp:
        try:
            write_otu_file(list(taxa), args.otu_fp)
        except RuntimeError as re:
            sys.exit(f"ERROR creating OTU file: \n\t{re}")

    if args.verbose:
        print(twdd(f"""
        BIOM-format table written to: {out_fp}
        Table contains {biomT.shape[0]} rows (OTUs) and {biomT.shape[1]} columns (Samples)
        and is {biomT.get_table_density():.1%} dense."""))


if __name__ == '__main__':
    main()

"""Fetch verified source bytes outside the evidence package without altering its seals."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,urllib.request
ROOT=Path(__file__).resolve().parent
def verify_bytes(path,size,expected):
    h=hashlib.sha256();n=0
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b);n+=len(b)
    if n!=size or h.hexdigest()!=expected:raise ValueError('Source size or SHA-256 mismatch')
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args()
    folder=args.directory.resolve()
    if folder==ROOT or ROOT in folder.parents:raise ValueError('Choose a new directory outside the evidence package')
    p=json.loads((ROOT/'SOURCE.json').read_text())
    folder.mkdir(parents=True,exist_ok=False)
    target=folder/Path(p['source_path']).name;partial=target.with_suffix('.partial')
    with urllib.request.urlopen(p['download_url'],timeout=120) as source,partial.open('xb') as out:
        if source.status!=200:raise ValueError('Unexpected HTTP status')
        for b in iter(lambda:source.read(2**20),b''):out.write(b)
    digest=verify_bytes(partial,p['source_bytes'],p['source_sha256']);partial.rename(target)
    receipt=dict(completed_utc=datetime.now(timezone.utc).isoformat(),source_url=p['download_url'],bytes=target.stat().st_size,sha256=digest,
                 scope='Reproduction of a previously used source; original source checksum verified; no new experiment.')
    (folder/'RAW_DOWNLOAD_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(str(target))
if __name__=='__main__':main()

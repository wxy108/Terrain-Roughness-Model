"""Verify every included terrain/geometry array against the data manifest."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    m=json.loads((ROOT/'data/manifest.json').read_text())
    records=m['records']+[m['full_base'],m['gait_bank']]
    for record in records:
        path=ROOT/record['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise SystemExit(f'Hash mismatch: {record["path"]}')
    print(f'Verified {len(records)} data files.')

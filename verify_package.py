"""Check every preserved file against the package manifest, not scientific validity."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text())
for name,value in manifest['files'].items():
    p=ROOT/name
    assert p.is_file() and p.stat().st_size==value['bytes'],name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==value['sha256'],name
print(json.dumps(dict(status='PASS',files=len(manifest['files']),scope='File integrity only')))

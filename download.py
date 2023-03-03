from pathlib import Path
from pydrive2.auth import GoogleAuth
from pydrive2.fs import GDriveFileSystem

CRED_FILE = '.credentials/gdrive.json'
GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = './credentials/client_secrets.json'


# https://stackoverflow.com/questions/24419188/automating-pydrive-verification-process
def authorize():
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile(CRED_FILE)
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile(CRED_FILE)
    return gauth


def remove_models(p: Path):
    if not p.exists(): return
    if p.is_file(): return p.unlink()
    for child in p.iterdir():
        remove_models(child)
    p.rmdir()


def download_models(out_dir="trained_agents", path='', root='root'):
    out_dir = Path(out_dir)
    remove_models(out_dir)
    out_dir.mkdir()

    fs = GDriveFileSystem(root, google_auth=authorize())
    assert fs.isdir(f"{root}/{path}")
    for remote in fs.glob(f"{root}/{path}/*.zip"):
        basename = remote.split('/')[-1]
        print(basename)
        local = str(out_dir / basename.replace(' ', '-'))
        # @TODO: Use `fs.info(...)` to only download if changed/newer?
        try:
            fs.download(remote, local)
        except Exception as e:
            print(f"Could not download {remote} ; message follows.")
            print(e)
    return




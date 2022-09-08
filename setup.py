data_path = 'data/raw/store.csv'
version = 'store_v1'
repo = 'https://github.com/degagawolde/pharmaceutical-sales-prediction.git'

import dvc.api
url = dvc.api.get_url(path=data_path,
                 repo=repo,
                 rev=version)
print(url)
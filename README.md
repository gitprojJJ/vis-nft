## Set-up virtual environment
One-time only


```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download data from google drive
- Download out.csv and network.csv from google drive and put the files in [/data](/data) folder 

## Activate virtual environment
Everytime working on the project

```sh
source .venv/bin/activate
```

## Loading pickled data (for fast loading)
Run [pickling.py](pickling.py) first to create pickled data. When there is pickled data, data_reader will use the pickled data.
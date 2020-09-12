import arrow
import h5py
import requests
from constants import DATA_URL, DATE_FORMAT, START_DATE, COUNTRY_IDS, OBLAST_IDS, COMBINED_IDS


def fetch_data_entry(dt: arrow.Arrow):
    params = {"to": dt.format(DATE_FORMAT)}
    response = requests.get(DATA_URL, params=params)

    if response.status_code == 200:
        return response.json()


def fetch_all_data():

    data = []
    for dt in arrow.Arrow.range("day", START_DATE, arrow.utcnow()):
        if resp := fetch_data_entry(dt):
            print(f"Adding date: " + dt.format(DATE_FORMAT))
            data.append(resp)
        else:
            print(f"Skipping date: " + dt.format(DATE_FORMAT))

    return data


def process_time_step(entry: list, regions: dict, keys: list = None):

    data = []

    for region_id in sorted(regions):
        row = [0 for _ in range(len(keys))]
        for entry_row in entry:
            if region_id == entry_row["id"]:
                row = [entry_row[key] for key in keys]

        data.append(row)

    return data


data = fetch_all_data()

keys = ["delta_existing", "delta_deaths", "delta_recovered"]

oblast_data = [process_time_step(entry["ukraine"], OBLAST_IDS, keys) for entry in data]
ukraine_data = [[sum(x) for x in zip(*entry)] for entry in oblast_data]

world_data = [process_time_step(entry["world"], COUNTRY_IDS, keys) for entry in data]
combined_data = [process_time_step(entry["world"] + entry["ukraine"], COMBINED_IDS, keys) for entry in data]


with h5py.File("dataset.h5", mode="w") as f:
    f.create_dataset("ukraine", data=ukraine_data)
    f.create_dataset("oblast", data=oblast_data)
    f.create_dataset("world", data=world_data)
    f.create_dataset("combined", data=combined_data)

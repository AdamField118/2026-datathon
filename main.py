import csv

if __name__ == "__main__":
    data_path = "data/team-stats"
    with open(f"{data_path}/advanced-stats.csv") as adv_file:
        advanced_stats = csv.reader(adv_file)
        for row in advanced_stats:
            print(row)

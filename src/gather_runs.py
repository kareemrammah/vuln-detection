#just collect every run into results/all_runs.json
import glob, json, os

runs={}
for f in sorted(glob.glob("results/*/seed*/metrics.json")):
    _, cfg, seed, _=os.path.normpath(f).split(os.sep)
    run=json.load(open(f))
    run["history"]=json.load(open(os.path.join(os.path.dirname(f), "history.json")))
    runs.setdefault(cfg, {})[seed]=run
json.dump(runs, open("results/all_runs.json", "w"), indent=1)
print(f"{sum(len(v) for v in runs.values())} runs in {len(runs)} configs -> results/all_runs.json")

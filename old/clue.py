import requests
import json

headers = { "user_key" : "f1f4250dc8c843043e79840e728b76cf" }

def get_landmarks():
    req_filter = {"where":{"l1000_type":"landmark"}}
    url = f"https://api.clue.io/api/genes?filter={json.dumps(req_filter)}"
    resp = requests.get(url=url,headers=headers)
    data = resp.json()
    print(data)

# req_filter = {"fields": ["pert_id","pert_iname", "structure_url", "canonical_smiles", "moa"],"where": {"pert_iname": {"inq": ["warfarin", "ambelline"]}}}
req_filter = {"where": {"pert_iname": {"inq": ["warfarin"]}}}
url = f"https://api.clue.io/api/perts?filter={json.dumps(req_filter)}"
resp = requests.get(url=url,headers=headers)
data = resp.json()
for d in data:
    for k in d:
        print(k,d[k])
    print()
    print()
    print()
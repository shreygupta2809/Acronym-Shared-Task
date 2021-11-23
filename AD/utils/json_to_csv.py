import json
import csv
 
def make_json(csvFilePath, jsonFilePath):
    data = []
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)
        for rows in csvReader:
            da = {}
            da['acronym'] = rows['acronym_']
            da['label'] = rows['expansion']
            da['ID'] = rows['id']
            da['sentence'] = rows['text']
            data.append(da)
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))
         
csvFilePath = r'./english/legal/update_train.csv'
jsonFilePath = r'./english/legal/update_train.json'
 
make_json(csvFilePath, jsonFilePath)

# with open("./english/scientific/output.json") as f:
#     data = json.load(f)

# # data = dict(data)
# # data = dict([(d['ID'], d['acronym']) for d in data])
# for k in data:
#     k['ID'] = str(k['ID'])

# with open("./english/scientific/update_train.json","w") as f:
#     json.dump(data,f,indent=2)

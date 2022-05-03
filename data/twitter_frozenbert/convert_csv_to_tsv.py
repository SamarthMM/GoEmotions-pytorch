import csv
import argparse

labels=[]
texts=[]

cli_parser=argparse.ArgumentParser()
cli_parser.add_argument("--file", type=str, required=True, help="name of csv file to convert")
args = cli_parser.parse_args()

with open(args.file,encoding='latin1') as f:
  csvreader=csv.reader(f)
  for i,line in enumerate(csvreader):
    labels.append(line[0])
    text=''
    for t in line[5:]:
      t=t.replace('\t', '  ')
      text+=t
    texts.append(text)

print(len(labels))
print(len(texts))

if (len(labels)!=len(texts)):
  print('n labels and texts dont match!')
else: 
  with open(args.file[:-3]+'tsv','w') as ff:
    tsv_writer = csv.writer(ff, delimiter='\t')
    for i in range(len(labels)):
      label=labels[i]
      if label=='2':
        continue
      if label=='4':
        label='1'
      tsv_writer.writerow([texts[i],label])
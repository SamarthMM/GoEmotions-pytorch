import csv

labels=[]
texts=[]

with open('test.csv',encoding='latin1') as f:
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
  with open('test.tsv','w',encoding='latin1') as ff:
    tsv_writer = csv.writer(ff, delimiter='\t')
    for i in range(len(labels)):
      if labels[i]=='2':
        continue
      if labels[i]=='4':
        label='3'
      elif labels[i]=='0':
        label='1'
      tsv_writer.writerow([texts[i],label])
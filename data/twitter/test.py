labels=[]
texts=[]

with open('training.1600000.processed.noemoticon.csv',encoding='latin1') as f:
  for line in f:
     line=line.strip().split(',')
     labels.append(int(line[0][1]))
     text=''
     for t in line[5:]:
       text+=t
     if text[0]=="\"":
       text=text[1:]
     if text[-1]=="\"":
       text=text[:-1]
     texts.append(text)
print(len(labels))
print(len(texts))
if (len(labels)!=len(texts)):
  print('n labels and texts dont match!')
else:
  with open('train_utf.tsv','w') as ff:
    for i in range(len(labels)):
      ff.write(texts[i]+'\t'+str(labels[i])+'\n')
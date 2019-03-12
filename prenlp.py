import json
import re
from dateutil.parser import parse
from nltk import tokenize

Date_pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December).+?(\d{1,2}),\s(\d\d\d\d)'
Name_pattern = r'^([a-z]{3})'

def tupletostr(t):
	string = ''
	for i in t:
		string = t[1]+' '+ t[0]+' ' + t[2]
	return string

import glob

filelist = []
for file in glob.glob("*.txt"):
    filelist.append(file)

# print(filelist)

def read_data(filename):
    with open(filename, "r") as f:
        text = f.read()
    return text


with open('hottest_words.json','r') as f:
	for line in f: # certain company certain date
		outputDict = {}
		date = ""
		CompanyName = ""
		text = ""
		dictionary = json.loads(line)
		for key in dictionary:
			keydate = re.findall(Date_pattern,key)
			Keyname = re.findall(Name_pattern,key)

			for d in keydate:
				d  = tupletostr(d)
				d = parse(d).strftime('%d/%m/%y')
				date += d
			
			for name in Keyname:
				CompanyName += name

			hotlist = dictionary[key]
			filename = key
			content_string = read_data(filename)
			content_string = tokenize.sent_tokenize(content_string)
			for sentence in content_string:
				for vocab in hotlist:
					if vocab in sentence:
						text += sentence
		outputDict['Date'] = date
		outputDict['CompanyName'] = CompanyName.upper()
		outputDict['Text'] = text
		with open ('8k.json','a') as f1:
		    json.dump(outputDict, f1)
		    f1.write('\n')
		f1.close()			
f.close()	
		

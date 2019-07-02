
# Scraping Twitter for Corporate Espionage
## (Using Webscraping, Facial Recognition and ML)

![headline](Images/headline.png "Twitter Project")


```python
#CREATE COLOR NUMBERS COLUMN

from IPython.display import Image, display
import urllib.request
from PIL import Image as img
import pandas as pd

df = pd.read_excel('twitter_followers_detailed.xlsx')
print('got df')
n = 0
for imageName in list(df["profile_image_url"]):
    try:
        urllib.request.urlretrieve(imageName, '/ml/out.png')
        colors = len(img.open("ml/out.png",mode="r").getcolors(maxcolors=9999999))
        df.at[n,"color_number"] = colors    
    except:
        df.at[n,"color_number"] = 0.01
    n+=1
    print(n)
```

    got df



```python
#Create Face Detection Column
import os
try:
    os.mkdir("ml")
except:
    print("folder already exists...")
    
import face_recognition

n = 0
for i in list(df["profile_image_url"]):
    try:
        urllib.request.urlretrieve(i, r'ml/out.png')
        detection = face_recognition.load_image_file("ml/out.png")
        if face_recognition.face_locations(detection):
            df.at[n,"face_detection"] = 1
        else:
            df.at[n,"face_detection"] = 0
    except:
        df.at[n,"face_detection"] = 999
    n+=1
```


```python
#Get rid of photos that did not connect and default photos
df = df[df.face_detection !=999.0]
df = df[df.color_number != 184]
```


```python
#URL Detect Column
array = []
for i in list(df["url"]):
    if str(i) != "nan":
        array.append(1)
    else:
        array.append(0)
df["url_detect"] = array
```

![headline](profile.png "profile")


```python
#output people labeled individuals to a csv/ then convert it to text for NLTK analysis
final = df

i_bios = final[final.label == "Individual"][['bio']]
i_bios.to_csv('i_bios.csv')

import csv
csv_file = (r'i_bios.csv')
txt_file = (r'i_bios.txt')
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
    my_output_file.close()
```


```python
#keywords function takes a filename, "grams" an integer meaning
#1-2- or 3 word n-grams and minimum count or appearances of the word or phrase

def keywords(file,grams,count):
    with open(str(file),'r',encoding="latin1") as myfile:
        my_string=myfile.read().replace('\n', '')

    string = ''.join(ch for ch in my_string if ch not in exclude)

    tokens = word_tokenize(string)
    text = nltk.Text(tokens)

    #array is the tuple of ngrams, array2 is the count of appearances, array1 is joined tuples, array 1 & 2 can be zipped into a dataframe
    array =[]
    array2 =[]
    bgs = nltk.ngrams(tokens,int(grams))
    fdist = nltk.FreqDist(bgs)
    for k,v in fdist.items():
        if v > int(count):
            array.append(k)
            array2.append(v)

    array1 = []
    for i in range(len(array)):
        x = ' '.join(map(str,array[i]))
        array1.append(x)

    df = pd.DataFrame({'phrase':array1,'count': array2}).sort_values(by="count",ascending=False)
    for i in list(df['phrase']):
        whitelist.append(i.lower())
    df.to_csv('output_'+str(grams)+'.csv')
```


```python
#import our dependancies, create empty whitelist array
#exclude includes ,$"@_[*|%)#+-<~^/;`=!:'&?}>({]\. (things to filter out)

import nltk
import string
from nltk import word_tokenize
import pandas as pd

whitelist = []
exclude = set(string.punctuation)
```


```python
#get 1grams, appearing more than 30 times,bigrams more than 20,
#and trigrams more than 10 times from the individuals bios
keywords('i_bios.txt',1,30)
keywords('i_bios.txt',2,20)
keywords('i_bios.txt',3,10)
```


```python
#remove stopwords from the individuals keywords array
from nltk.corpus import stopwords

new_whitelist = []

stop = list(set(stopwords.words('english')))
additional_stop = ["business"]
stop += additional_stop
safewords = ["I"]
stop = [i for i in stop if i not in safewords]

for i in whitelist:
    try:
        if (i.split(" ")[0] not in stop) & (i.split(" ")[1] not in stop):
            new_whitelist.append(i)
    except:
        if i not in stop:
            new_whitelist.append(i)
        pass
```


```python
#remove dupes, filter non-alpha keywords and save it to individual_keywords variable
new_whitelist = list(set(new_whitelist))

for i in range(len(new_whitelist)):
    try:
        if new_whitelist[i].split(" ")[0].isalpha() == False:
            new_whitelist.pop(i)
    except:
        pass
    
individual_keywords = new_whitelist
```

![headline](individual_keywords.png "Individual Keywords")


```python
#reset our variables and do all the same for business keywords
new_whitelist = []
whitelist = []
```


```python
b_bios = final[final.label == "Business"][["bio"]]
b_bios.to_csv('b_bios.csv')
csv_file = (r'b_bios.csv')
txt_file = (r'b_bios.txt')
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r") as my_input_file:
        [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
    my_output_file.close()
```


```python
keywords('b_bios.txt',1,30)
keywords('b_bios.txt',2,20)
keywords('b_bios.txt',3,10)
```


```python
stop = list(set(stopwords.words('english')))
additional_stop = ["business","us"]
stop += additional_stop
safewords = ["I"]
stop = [i for i in stop if i not in safewords]

for i in whitelist:
    try:
        if (i.split(" ")[0] not in stop) & (i.split(" ")[1] not in stop):
            new_whitelist.append(i)
    except:
        if i not in stop:
            new_whitelist.append(i)
        pass
    
new_whitelist = list(set(new_whitelist))

# filter non-alpha stuff
for i in range(len(new_whitelist)):
    if new_whitelist[i].split(" ")[0].isalpha() == False:
        new_whitelist.pop(i)
        
business_keywords = new_whitelist
```

![headline](business_keywords.png "Individual Keywords")


```python
business_keywords
```




    ['platform',
     'online',
     'services',
     'small',
     'financial',
     'capital',
     'lending',
     'company',
     'provide',
     'funding',
     'help',
     'loans',
     'small businesses',
     'solutions',
     'finance',
     'businesses']




```python
individual_keywords
```




    ['husband',
     'financial',
     'capital',
     'investor',
     'husband father',
     'ceo',
     'marketing',
     'cofounder',
     'father',
     'founder ceo',
     'fintech',
     'entrepreneur',
     'founder']




```python
# Create keyword counter columns for individual keywords, and business keywords
i_nums = []
for x in range(len(final)):
    test = list(final["bio"])[x]
    n = 0
    try:
        for i in individual_keywords:
            if i in test:
                n+=1
        i_nums.append(n)
    except:
        i_nums.append(0)
```


```python
b_nums = []
for x in range(len(final)):
    test = list(final["bio"])[x]
    n = 0
    try:
        for i in business_keywords:
            if i in test:
                n+=1
        b_nums.append(n)
    except:
        b_nums.append(0)
```


```python
final["b_key_count"] = b_nums
final["i_key_count"] = i_nums
```


```python
final[["bio","b_key_count","i_key_count"]].sort_values(by="b_key_count",ascending=False).head(40)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bio</th>
      <th>b_key_count</th>
      <th>i_key_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1535</th>
      <td>We provide working capital solutions to small ...</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2170</th>
      <td>LendVantage helps small businesses find loans,...</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2267</th>
      <td>Global Lending is committed to helping small b...</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1698</th>
      <td>#Onlinelending company empowering #smallbusine...</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2204</th>
      <td>Argentum Capital Group, LLC is a finance compa...</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>824</th>
      <td>We specialise in business loans for small busi...</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>742</th>
      <td>We provide small businesses with quick and eas...</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1324</th>
      <td>AMP Advance is a Miami based industry leading ...</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2117</th>
      <td>Fundingportal is a funding services and soluti...</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1778</th>
      <td>Arrington Accounting Srvcs provides experience...</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1091</th>
      <td>We are here to educate, empower and enable sma...</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1540</th>
      <td>We support alternative finance companies that ...</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>922</th>
      <td>Your full service lender for small business lo...</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2074</th>
      <td>Legend Funding is comprised of financial exper...</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1210</th>
      <td>Business Funding Relations is not a bank, it i...</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1187</th>
      <td>Our innovative online credit solutions provide...</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2410</th>
      <td>Oxygen Funding, Inc. is an invoice factoring  ...</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1515</th>
      <td>We help small-to-enterprise level businesses g...</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1904</th>
      <td>SeedInvest is a leading equity crowdfunding pl...</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1905</th>
      <td>Invoice your customer √¢‚Ç¨¬¢ Fund invoice √¢‚...</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1498</th>
      <td>Website helping small businesses find the perf...</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>263</th>
      <td>We provide simple, accurate &amp; affordable #payr...</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>845</th>
      <td>Cloud-based provider of human capital manageme...</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1494</th>
      <td>Arcarius provides short-term working capital a...</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1652</th>
      <td>PrestaCap is an innovative financing platform ...</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1727</th>
      <td>Acquiring small businesses, commercializing ne...</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>913</th>
      <td>We're a business finance company. Since 2011, ...</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1737</th>
      <td>We are specialists in helping insurers and fin...</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1544</th>
      <td>India's leading #smallbusiness #digital #lendi...</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>747</th>
      <td>We are transforming how small businesses acces...</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>735</th>
      <td>#MCAleads World Solutions is a international o...</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1874</th>
      <td>We help small businesses access fast cash with...</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1394</th>
      <td>We are a company dedicated to find term loans ...</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1072</th>
      <td>The Strategic Hub, Inc. is a boutique firm tha...</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2162</th>
      <td>Funding Tower provides immediate business loan...</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2142</th>
      <td>Providing America with job opportunity and ass...</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>862</th>
      <td>Lendified provides fast, easy, and affordable ...</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1933</th>
      <td>Since 1989, MIAC has been the leading provider...</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1343</th>
      <td>#FinTech company provides investment professio...</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1523</th>
      <td>Leading news source for all things alternative...</td>
      <td>4</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
final[["bio","b_key_count","i_key_count"]].sort_values(by="i_key_count",ascending=False).head(40)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bio</th>
      <th>b_key_count</th>
      <th>i_key_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2148</th>
      <td>father of 4, husband, outdoor guy, entrepreneu...</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2242</th>
      <td>Lifetime entrepreneur, serial founder and inve...</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1478</th>
      <td>David lives in Brooklyn &amp; is a husband, father...</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>406</th>
      <td>We√¢‚Ç¨‚Ñ¢re mutifaceted entrepreneurs, financ...</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1912</th>
      <td>CEO &amp; Founder of ConveyIQ (http://www.conveyiq...</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2131</th>
      <td>British/Canadian in London. Co-founder of @SAS...</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1687</th>
      <td>Venture investor, fintech afficionado, feminis...</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2130</th>
      <td>Cofounder &amp; CEO of @P2Binvestor a crowdlending...</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1292</th>
      <td>Strategic marketing communications specialist ...</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2188</th>
      <td>Manager, Media Sales @expedia / marketing pro ...</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1895</th>
      <td>husband father and grandfather</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1549</th>
      <td>ex-ceo &amp; co-founder suit guy@georgeandking, bu...</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1926</th>
      <td>Co-founder of Neyber @helloneyber, Trustee @St...</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2241</th>
      <td>Digital marketing manager at @TransparentBPO; ...</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2524</th>
      <td>CEO of OnDeck, husband, father, entrepreneur, ...</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>@drtharon smith: economist entrepreneur focuse...</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1969</th>
      <td>CEO of BridgePoint Capital, husband, father, e...</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2109</th>
      <td>Father, husband, entrepreneur and investor. Mo...</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1219</th>
      <td>Founder of Beyond Payment Systems and Merchant...</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>744</th>
      <td>The only direct real estate investment platfor...</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2269</th>
      <td>Co-Founder @Logic_FI. 2x #fintech entrepreneur...</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1550</th>
      <td>CEO &amp; Co-Founder of @FundingWizards - a market...</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2447</th>
      <td>Founder/CEO of Bizignition &amp; SIPN, entrepreneu...</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2155</th>
      <td>CFO @VOLOCOPTER, 9ys father, 19ys entrepreneur...</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2254</th>
      <td>We are #FinTech thought leaders connecting inn...</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1602</th>
      <td>#FinTech Co-founder at @banknovo. Passionate a...</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>marketer, experience seeker, skier/biker/hiker...</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>Wayfarer, husband, father, CEO, actor, playwri...</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2081</th>
      <td>Business owner, husband, father, world travell...</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2311</th>
      <td>Business owner, Chargers fanatic, husband, and...</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1741</th>
      <td>Fintech entrepreneur and SME enthusiast. CEO a...</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>782</th>
      <td>Dad, husband, CEO and founder of @lendingworks</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2461</th>
      <td>teacher √¢‚Ç¨¬¢life coach √¢‚Ç¨¬¢ photographer...</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1475</th>
      <td>I enjoy all things about #branding #marketing....</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1931</th>
      <td>father,Husband,Entrepreneur, and Realestate in...</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>733</th>
      <td>Marketplace lending entrepreneur, industry obs...</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2285</th>
      <td>husband, father, brother, friend</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1965</th>
      <td>Lucky husband &amp; father of three, runner, and a...</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>278</th>
      <td>#Smallbiz, #equipmentfinance, #entrepreneur #t...</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2405</th>
      <td>Your daily decisions determine your direction ...</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
final.to_csv('final_output.csv',encoding="UTF-8")
```

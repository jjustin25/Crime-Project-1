

```python
%matplotlib notebook
```


```python
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
```


```python
indir="cp16sb/"
outfile="../out/Concatenated.csv"

os.chdir(indir)
fileList=glob.glob("*.csv")
dfList=[]
for filename in fileList:
    print(filename)
    df=pd.read_csv(filename, header=None,encoding='cp1252')
    dfList.append(df)
concatDF=pd.concat(dfList, axis=0)
concatDF.to_csv(outfile, index=None)
```

    cp16sbat01.csv
    cp16sbat02.csv
    cp16sbat03.csv
    cp16sbat04.csv
    cp16sbat05.csv
    cp16sbat06.csv
    cp16sbat07.csv
    cp16sbat08.csv
    cp16sbf01.csv
    cp16sbf02.csv
    cp16sbf03.csv
    cp16sbf04.csv
    cp16sbf05.csv
    cp16sbt01.csv
    cp16sbt02.csv
    cp16sbt03.csv
    cp16sbt04.csv
    


```python
concatDF.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bureau of Justice Statistics</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Filename: cp16sbat01.csv</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Appendix table 1. Capital offenses, by state, ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Report title: Capital Punishment, 2016 - Stati...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Data source: National Prisoner Statistics prog...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
violent_crime = pd.read_csv("../serial_database.csv")
```

    C:\Users\jjust\Anaconda3\envs\PythonData\lib\site-packages\IPython\core\interactiveshell.py:2728: DtypeWarning: Columns (16) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)
    


```python
violent_crime.head(15)
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
      <th>Record ID</th>
      <th>Agency Code</th>
      <th>Agency Name</th>
      <th>Agency Type</th>
      <th>City</th>
      <th>State</th>
      <th>Year</th>
      <th>Month</th>
      <th>Incident</th>
      <th>Crime Type</th>
      <th>...</th>
      <th>Victim Ethnicity</th>
      <th>Perpetrator Sex</th>
      <th>Perpetrator Age</th>
      <th>Perpetrator Race</th>
      <th>Perpetrator Ethnicity</th>
      <th>Relationship</th>
      <th>Weapon</th>
      <th>Victim Count</th>
      <th>Perpetrator Count</th>
      <th>Record Source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>January</td>
      <td>1</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Male</td>
      <td>15</td>
      <td>Native American/Alaska Native</td>
      <td>Unknown</td>
      <td>Acquaintance</td>
      <td>Blunt Object</td>
      <td>0</td>
      <td>0</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>March</td>
      <td>1</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Male</td>
      <td>42</td>
      <td>White</td>
      <td>Unknown</td>
      <td>Acquaintance</td>
      <td>Strangulation</td>
      <td>0</td>
      <td>0</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>March</td>
      <td>2</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>0</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>0</td>
      <td>0</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>April</td>
      <td>1</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Male</td>
      <td>42</td>
      <td>White</td>
      <td>Unknown</td>
      <td>Acquaintance</td>
      <td>Strangulation</td>
      <td>0</td>
      <td>0</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>April</td>
      <td>2</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>0</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>0</td>
      <td>1</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>May</td>
      <td>1</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Male</td>
      <td>36</td>
      <td>White</td>
      <td>Unknown</td>
      <td>Acquaintance</td>
      <td>Rifle</td>
      <td>0</td>
      <td>0</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>May</td>
      <td>2</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Male</td>
      <td>27</td>
      <td>Black</td>
      <td>Unknown</td>
      <td>Wife</td>
      <td>Knife</td>
      <td>0</td>
      <td>0</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>June</td>
      <td>1</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Male</td>
      <td>35</td>
      <td>White</td>
      <td>Unknown</td>
      <td>Wife</td>
      <td>Knife</td>
      <td>0</td>
      <td>0</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>June</td>
      <td>2</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>0</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Firearm</td>
      <td>0</td>
      <td>0</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>June</td>
      <td>3</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Male</td>
      <td>40</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Firearm</td>
      <td>0</td>
      <td>1</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>July</td>
      <td>1</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>0</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>0</td>
      <td>1</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>July</td>
      <td>2</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Male</td>
      <td>49</td>
      <td>White</td>
      <td>Unknown</td>
      <td>Stranger</td>
      <td>Shotgun</td>
      <td>0</td>
      <td>0</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>July</td>
      <td>3</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Male</td>
      <td>39</td>
      <td>Black</td>
      <td>Unknown</td>
      <td>Girlfriend</td>
      <td>Blunt Object</td>
      <td>0</td>
      <td>0</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>August</td>
      <td>1</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Male</td>
      <td>49</td>
      <td>White</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Fall</td>
      <td>0</td>
      <td>0</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>August</td>
      <td>2</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>0</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Handgun</td>
      <td>0</td>
      <td>0</td>
      <td>FBI</td>
    </tr>
  </tbody>
</table>
<p>15 rows × 24 columns</p>
</div>




```python
violent_crime.isnull().values.any()
```




    False




```python
violent_crime.rename(columns={'Perpetrator Age':'Perpetrator_Age'}, inplace=True)
violent_crime.head()
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
      <th>Record ID</th>
      <th>Agency Code</th>
      <th>Agency Name</th>
      <th>Agency Type</th>
      <th>City</th>
      <th>State</th>
      <th>Year</th>
      <th>Month</th>
      <th>Incident</th>
      <th>Crime Type</th>
      <th>...</th>
      <th>Victim Ethnicity</th>
      <th>Perpetrator Sex</th>
      <th>Perpatrator_Age</th>
      <th>Perpetrator Race</th>
      <th>Perpetrator Ethnicity</th>
      <th>Relationship</th>
      <th>Weapon</th>
      <th>Victim Count</th>
      <th>Perpetrator Count</th>
      <th>Record Source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>January</td>
      <td>1</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Male</td>
      <td>15</td>
      <td>Native American/Alaska Native</td>
      <td>Unknown</td>
      <td>Acquaintance</td>
      <td>Blunt Object</td>
      <td>0</td>
      <td>0</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>March</td>
      <td>1</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Male</td>
      <td>42</td>
      <td>White</td>
      <td>Unknown</td>
      <td>Acquaintance</td>
      <td>Strangulation</td>
      <td>0</td>
      <td>0</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>March</td>
      <td>2</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>0</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>0</td>
      <td>0</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>April</td>
      <td>1</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Male</td>
      <td>42</td>
      <td>White</td>
      <td>Unknown</td>
      <td>Acquaintance</td>
      <td>Strangulation</td>
      <td>0</td>
      <td>0</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>April</td>
      <td>2</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>0</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>0</td>
      <td>1</td>
      <td>FBI</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
#in place dropna unknown for perp age
```


```python
#numpy
p1=25
p2=50
p3=75
q1 = np.percentile(violent_crime['Perpatrator_Age'],  p1)
q2 = np.percentile(violent_crime['Perpatrator_Age'],  p2)
q3 = np.percentile(violent_crime['Perpatrator_Age'],  p3)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-47-566acc4bbb2c> in <module>()
          3 p2=50
          4 p3=75
    ----> 5 q1 = np.percentile(violent_crime['Perpatrator_Age'],  p1)
          6 q2 = np.percentile(violent_crime['Perpatrator_Age'],  p2)
          7 q3 = np.percentile(violent_crime['Perpatrator_Age'],  p3)
    

    ~\Anaconda3\envs\PythonData\lib\site-packages\numpy\lib\function_base.py in percentile(a, q, axis, out, overwrite_input, interpolation, keepdims)
       4289     r, k = _ureduce(a, func=_percentile, q=q, axis=axis, out=out,
       4290                     overwrite_input=overwrite_input,
    -> 4291                     interpolation=interpolation)
       4292     if keepdims:
       4293         return r.reshape(q.shape + k)
    

    ~\Anaconda3\envs\PythonData\lib\site-packages\numpy\lib\function_base.py in _ureduce(a, func, **kwargs)
       4031         keepdim = (1,) * a.ndim
       4032 
    -> 4033     r = func(a, **kwargs)
       4034     return r, keepdim
       4035 
    

    ~\Anaconda3\envs\PythonData\lib\site-packages\numpy\lib\function_base.py in _percentile(a, q, axis, out, overwrite_input, interpolation, keepdims)
       4390         weights_above.shape = weights_shape
       4391 
    -> 4392         ap.partition(concatenate((indices_below, indices_above)), axis=axis)
       4393 
       4394         # ensure axis with qth is first
    

    TypeError: '<' not supported between instances of 'str' and 'int'


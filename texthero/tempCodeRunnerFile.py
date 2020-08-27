import texthero as hero
import pandas as pd
df = pd.read_csv(
    "https://raw.githubusercontent.com/jbesomi/texthero/master/dataset/bbcsport.csv"
)
hero.show(df)

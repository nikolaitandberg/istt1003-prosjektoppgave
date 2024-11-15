import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

df = pd.read_csv("Data/lego.population.csv", sep = ",", encoding = "latin1")

# fjerner forklaringsvariabler vi ikke trenger
df2 = df[['Set_Name', 'Theme', 'Pieces', 'Price', 'Pages',  'Unique_Pieces']]

# fjerner observasjoner med manglende datapunkter
df2 = df2.dropna()

# gjør themes om til string og fjern alle tegn vi ikke vil ha med
df2['Theme'] = df2['Theme'].astype(str)
df2['Theme'] = df2['Theme'].str.replace(r'[^a-zA-Z0-9\s-]', '', regex = True)

# fjerner dollartegn og trademark-tegn fra datasettet
df2['Price'] = df2['Price'].str.replace('$', '', regex = False)

# og gjør så prisen om til float
df2['Price'] = df2['Price'].astype(float)

# Definer kategoriene og temaene som tilhører hver kategori
categories = {
    'Barn': ['DOTS', 'DUPLO', 'Disney', 'Juniors', 'LEGO Frozen 2', 'Minions', 'Powerpuff Girls', 'Trolls World Tour', 'Unikitty','Friends', 'City'], #0 - 9
    'Ungdom': ['Batman', 'BrickHeadz', 'Creator 3-in-1', 'Hidden Side', 'Minecraft', 'Monkie Kid', 'Overwatch', 'Spider-Man', 'THE LEGO MOVIE 2', 'Harry Potter', 'NINJAGO'], # 9 - 15
    'Voksen': ['Architecture', 'Classic', 'Creator Expert', 'DC', 'Ideas', 'Jurassic World', 'Marvel', 'Minifigures', 'Powered UP', 'Speed Champions', 'Stranger Things', 'Technic', 'Xtra','Star Wars']  # 15+
}

# Funksjon for å bestemme kategori basert på tema
def get_category(theme):
    for category, themes in categories.items():
        if theme in themes:
            return category
    return 'Voksen'

# Legg til en ny kolonne 'cat' basert på tema
df2['cat'] = df2['Theme'].apply(get_category)


#  TODO: endre til kategoriene over istedenfor
resultater = []
plt.figure(figsize=(10, 5))
for i, cat in enumerate(categories):
    # MLR
    modell3 = smf.ols('Price ~ Pieces' , data = df2[df2['cat'].isin([cat])])
    resultater.append(modell3.fit())
    print(resultater[i].summary())
    print(f"Kategori: {cat}")
    print(resultater[i].params[...])
    #print(resultater[i].params["R-squared"])

    # RESIDUALS
    redidual_data = smf.ols('Price ~ Pieces*Theme' , data = df2[df2['cat'].isin([cat])])
    redidual_data.fit()

    sns.scatterplot(x = redidual_data.fit().fittedvalues, y = redidual_data.fit().resid, color=plt.cm.tab10(i), label=cat)
plt.title("Residualplott")
plt.ylabel("Residual")
plt.xlabel("Predikert verdi")
plt.legend()
plt.grid()
plt.show()

redidual_data = smf.ols('Price ~ Pieces*Theme' , data = df2)
redidual_data.fit()
sm.qqplot(redidual_data.fit().resid, line = '45', fit = True, ax=plt.gca())
plt.title("QQ-plot")
plt.xlabel("Kvantiler i residualene")
plt.ylabel("Kvantiler i normalfordelingen")
plt.legend()
plt.grid()
plt.show()

# #  plott av dataene og regresjonslinjene
for i, cat in enumerate(categories):
    plt.scatter(df2[df2['cat'].isin([cat])]['Pieces'], df2[df2['cat'].isin([cat])]['Price'], color=plt.cm.tab10(i))

    slope = resultater[i].params['Pieces'] # stigningstallet a
    intercept = resultater[i].params['Intercept'] # konstantleddet b

    regression_x = np.array(df2[df2['cat'].isin([cat])]['Pieces'])
    regression_y = slope * regression_x + intercept
    plt.plot(regression_x, regression_y, color=plt.cm.tab10(i), label=cat)

    
plt.xlabel('Antall brikker')
plt.ylabel('Pris')
plt.title('Kryssplott')
plt.legend()
plt.grid()
plt.show()
plt.savefig("output/oppg.png")

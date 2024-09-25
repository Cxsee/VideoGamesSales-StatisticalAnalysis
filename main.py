import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns 
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from scipy.stats import shapiro, t
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings  

warnings.filterwarnings('ignore', category=FutureWarning)



#Pulizia e preparazione dataset

dataset=pd.read_csv(".\Dataset\Video_Games_Sales.csv")
dataset= dataset.rename(columns={
    'Name': 'Nome',
    'Platform': 'Piattaforma',
    'Year_of_Release': 'Anno_di_uscita',
    'Genre': 'Genere',
    'Publisher': 'Editore',
    'NA_Sales': 'Vendite_NA',
    'EU_Sales': 'Vendite_EU',
    'JP_Sales': 'Vendite_JP',
    'Other_Sales': 'Altre_Vendite',
    'Global_Sales': 'Vendite_Globali',
    'Critic_Score': 'Punteggio_Critici',
    'Critic_Count': 'Numero_Critici',
    'User_Score': 'Punteggio_Utenti',
    'User_Count': 'Numero_Utenti',
    'Developer': 'Sviluppatore',
    'Rating': 'Classificazione'
})
#print(dataset.head())


#print(dataset.isnull().sum())

# Rimuovere le righe con valori NaN
dataset_cleaned = dataset.dropna()





dataset_cleaned=dataset_cleaned.drop(columns=["Nome","Editore","Sviluppatore"])

#Trasformo i dati object in dati numerici
dataset_cleaned['Punteggio_Critici'] = pd.to_numeric(dataset['Punteggio_Critici'], errors='coerce')
dataset_cleaned['Punteggio_Utenti'] = pd.to_numeric(dataset['Punteggio_Utenti'], errors='coerce')
dataset_cleaned['Punteggio_Utenti']=dataset_cleaned['Punteggio_Utenti'].astype(float)

dataset_cleaned = dataset_cleaned.dropna()



#################################################################################################################
###############################################EDA###############################################################
#################################################################################################################
#################################################################################################################


#Matrice di Correlazione
numeric_columns = dataset_cleaned.select_dtypes(include=['number'])

corr_matrix = numeric_columns.corr()


plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matrice di correlazione')
plt.show()





#################################################################################################################
##################################VENDITE GLOBALI PER GENERE,####################################################
########################ANNO,CRITIC E USER SCORE, PIATTAFORMA, ESRB RATE#########################################
#################################################################################################################

# Calcola le vendite totali per genere
vendite_per_genere = dataset_cleaned.groupby('Genere')['Vendite_Globali'].sum()

# Calcola le vendite totali per anno
vendite_per_anno = dataset_cleaned.groupby('Anno_di_uscita')['Vendite_Globali'].sum()



# Calcola le vendite totali per piattaforma
vendite_per_piattaforma = dataset_cleaned.groupby('Piattaforma')['Vendite_Globali'].sum()

# Calcola le vendite totali per voti ESBR
vendite_per_esrb = dataset_cleaned.groupby('Classificazione')['Vendite_Globali'].sum()

# Creazione della figura e degli assi per la griglia 3x2
fig, axs = plt.subplots(3, 2, figsize=(15, 12))

# Vendite per piattaforma
axs[0, 0].bar(vendite_per_piattaforma.index, vendite_per_piattaforma.values, color='skyblue')
axs[0, 0].set_title('Vendite Globali per Piattaforma')
axs[0, 0].set_xlabel('Piattaforma')
axs[0, 0].set_ylabel('Vendite Globali (Millions)')
axs[0, 0].tick_params(axis='x', rotation=45)
axs[0, 0].grid(True)

# Vendite per genere
axs[0, 1].bar(vendite_per_genere.index, vendite_per_genere.values, color='skyblue')
axs[0, 1].set_title('Vendite Globali per Genere')
axs[0, 1].set_xlabel('Genere')
axs[0, 1].set_ylabel('Vendite Globali (Millions)')
axs[0, 1].tick_params(axis='x', rotation=45)
axs[0, 1].grid(True)

# Vendite per anno
axs[1, 0].bar(vendite_per_anno.index, vendite_per_anno.values, color='skyblue')
axs[1, 0].set_title('Vendite Globali per Anno')
axs[1, 0].set_xlabel('Anno')
axs[1, 0].set_ylabel('Vendite Globali (Millions)')
axs[1, 0].tick_params(axis='x', rotation=90)
axs[1, 0].grid(True)

# Vendite per Publisher
axs[1, 1].bar(vendite_per_esrb.index, vendite_per_esrb.values, color='skyblue')
axs[1, 1].set_title('Vendite Globali per Voti ESRB')
axs[1, 1].set_xlabel('Voti ESRB')
axs[1, 1].set_ylabel('Vendite Globali (Millions)')
axs[1, 1].tick_params(axis='x', rotation=90)
axs[1, 1].grid(True)

# Vendite vs Critic Score
axs[2, 0].scatter(dataset_cleaned['Punteggio_Critici'], dataset_cleaned['Vendite_Globali'], color='skyblue', alpha=0.5)
axs[2, 0].set_title('Vendite Globali vs Critic Score')
axs[2, 0].set_xlabel('Critic Score')
axs[2, 0].set_ylabel('Vendite Globali (Millions)')
axs[2, 0].grid(True)

# Vendite vs User Score
axs[2, 1].scatter(dataset_cleaned['Punteggio_Utenti'], dataset_cleaned['Vendite_Globali'], color='skyblue', alpha=0.5)
axs[2, 1].set_title('Vendite Globali vs User Score')
axs[2, 1].set_xlabel('User Score')
axs[2, 1].set_ylabel('Vendite Globali (Millions)')
axs[2, 1].grid(True)

# Aggiunge layout stretto per migliorare la spaziatura
plt.tight_layout()

# Mostra il grafico
plt.show()



#################################################################################################################
####################################VENDITE EU PER GENERE,#######################################################
########################ANNO,CRITIC E USER SCORE, PIATTAFORMA, ESRB RATE#########################################
#################################################################################################################

# Calcola le vendite totali per genere
vendite_per_genere = dataset_cleaned.groupby('Genere')['Vendite_EU'].sum()

# Calcola le vendite totali per anno
vendite_per_anno = dataset_cleaned.groupby('Anno_di_uscita')['Vendite_EU'].sum()

# Calcola le vendite totali per piattaforma
vendite_per_piattaforma = dataset_cleaned.groupby('Piattaforma')['Vendite_EU'].sum()

# Calcola le vendite totali per voti ESBR
vendite_per_esrb = dataset_cleaned.groupby('Classificazione')['Vendite_EU'].sum()

# Creazione della figura e degli assi per la griglia 3x2
fig, axs = plt.subplots(3, 2, figsize=(15, 12))

# Vendite per piattaforma
axs[0, 0].bar(vendite_per_piattaforma.index, vendite_per_piattaforma.values, color='skyblue')
axs[0, 0].set_title('Vendite EU per Piattaforma')
axs[0, 0].set_xlabel('Piattaforma')
axs[0, 0].set_ylabel('Vendite EU (Millions)')
axs[0, 0].tick_params(axis='x', rotation=45)
axs[0, 0].grid(True)

# Vendite per genere
axs[0, 1].bar(vendite_per_genere.index, vendite_per_genere.values, color='skyblue')
axs[0, 1].set_title('Vendite EU per Genere')
axs[0, 1].set_xlabel('Genere')
axs[0, 1].set_ylabel('Vendite EU (Millions)')
axs[0, 1].tick_params(axis='x', rotation=45)
axs[0, 1].grid(True)

# Vendite per anno
axs[1, 0].bar(vendite_per_anno.index, vendite_per_anno.values, color='skyblue')
axs[1, 0].set_title('Vendite EU per Anno')
axs[1, 0].set_xlabel('Anno')
axs[1, 0].set_ylabel('Vendite EU (Millions)')
axs[1, 0].tick_params(axis='x', rotation=90)
axs[1, 0].grid(True)

# Vendite per Publisher
axs[1, 1].bar(vendite_per_esrb.index, vendite_per_esrb.values, color='skyblue')
axs[1, 1].set_title('Vendite EU per Voti ESRB')
axs[1, 1].set_xlabel('Voti ESRB')
axs[1, 1].set_ylabel('Vendite EU (Millions)')
axs[1, 1].tick_params(axis='x', rotation=90)
axs[1, 1].grid(True)

# Vendite vs Critic Score
axs[2, 0].scatter(dataset_cleaned['Punteggio_Critici'], dataset_cleaned['Vendite_EU'], color='skyblue', alpha=0.5)
axs[2, 0].set_title('Vendite EU vs Critic Score')
axs[2, 0].set_xlabel('Critic Score')
axs[2, 0].set_ylabel('Vendite EU (Millions)')
axs[2, 0].grid(True)

# Vendite vs User Score
axs[2, 1].scatter(dataset_cleaned['Punteggio_Utenti'], dataset_cleaned['Vendite_EU'], color='skyblue', alpha=0.5)
axs[2, 1].set_title('Vendite EU vs User Score')
axs[2, 1].set_xlabel('User Score')
axs[2, 1].set_ylabel('Vendite EU (Millions)')
axs[2, 1].grid(True)

# Aggiunge layout stretto per migliorare la spaziatura
plt.tight_layout()

# Mostra il grafico
plt.show()


#################################################################################################################
####################################VENDITE NA PER GENERE,#######################################################
########################ANNO,CRITIC E USER SCORE, PIATTAFORMA, ESRB RATE#########################################
#################################################################################################################

# Calcola le vendite totali per genere
vendite_per_genere = dataset_cleaned.groupby('Genere')['Vendite_NA'].sum()

# Calcola le vendite totali per anno
vendite_per_anno = dataset_cleaned.groupby('Anno_di_uscita')['Vendite_NA'].sum()

# Calcola le vendite totali per piattaforma
vendite_per_piattaforma = dataset_cleaned.groupby('Piattaforma')['Vendite_NA'].sum()

# Calcola le vendite totali per voti ESBR
vendite_per_esrb = dataset_cleaned.groupby('Classificazione')['Vendite_NA'].sum()

# Creazione della figura e degli assi per la griglia 3x2
fig, axs = plt.subplots(3, 2, figsize=(15, 12))

# Vendite per piattaforma
axs[0, 0].bar(vendite_per_piattaforma.index, vendite_per_piattaforma.values, color='skyblue')
axs[0, 0].set_title('Vendite NA per Piattaforma')
axs[0, 0].set_xlabel('Piattaforma')
axs[0, 0].set_ylabel('Vendite NA (Millions)')
axs[0, 0].tick_params(axis='x', rotation=45)
axs[0, 0].grid(True)

# Vendite per genere
axs[0, 1].bar(vendite_per_genere.index, vendite_per_genere.values, color='skyblue')
axs[0, 1].set_title('Vendite NA per Genere')
axs[0, 1].set_xlabel('Genere')
axs[0, 1].set_ylabel('Vendite NA (Millions)')
axs[0, 1].tick_params(axis='x', rotation=45)
axs[0, 1].grid(True)

# Vendite per anno
axs[1, 0].bar(vendite_per_anno.index, vendite_per_anno.values, color='skyblue')
axs[1, 0].set_title('Vendite NA per Anno')
axs[1, 0].set_xlabel('Anno')
axs[1, 0].set_ylabel('Vendite NA (Millions)')
axs[1, 0].tick_params(axis='x', rotation=90)
axs[1, 0].grid(True)

# Vendite per Publisher
axs[1, 1].bar(vendite_per_esrb.index, vendite_per_esrb.values, color='skyblue')
axs[1, 1].set_title('Vendite NA per Voti ESRB')
axs[1, 1].set_xlabel('Voti ESRB')
axs[1, 1].set_ylabel('Vendite NA (Millions)')
axs[1, 1].tick_params(axis='x', rotation=90)
axs[1, 1].grid(True)

# Vendite vs Critic Score
axs[2, 0].scatter(dataset_cleaned['Punteggio_Critici'], dataset_cleaned['Vendite_NA'], color='skyblue', alpha=0.5)
axs[2, 0].set_title('Vendite NA vs Critic Score')
axs[2, 0].set_xlabel('Critic Score')
axs[2, 0].set_ylabel('Vendite NA (Millions)')
axs[2, 0].grid(True)

# Vendite vs User Score
axs[2, 1].scatter(dataset_cleaned['Punteggio_Utenti'], dataset_cleaned['Vendite_NA'], color='skyblue', alpha=0.5)
axs[2, 1].set_title('Vendite NA vs User Score')
axs[2, 1].set_xlabel('User Score')
axs[2, 1].set_ylabel('Vendite NA (Millions)')
axs[2, 1].grid(True)

# Aggiunge layout stretto per migliorare la spaziatura
plt.tight_layout()

# Mostra il grafico
plt.show()

#################################################################################################################
####################################VENDITE JP PER GENERE,#######################################################
########################ANNO,CRITIC E USER SCORE, PIATTAFORMA, ESRB RATE#########################################
#################################################################################################################

# Calcola le vendite totali per genere
vendite_per_genere = dataset_cleaned.groupby('Genere')['Vendite_JP'].sum()

# Calcola le vendite totali per anno
vendite_per_anno = dataset_cleaned.groupby('Anno_di_uscita')['Vendite_JP'].sum()

# Calcola le vendite totali per piattaforma
vendite_per_piattaforma = dataset_cleaned.groupby('Piattaforma')['Vendite_JP'].sum()

# Calcola le vendite totali per voti ESBR
vendite_per_esrb = dataset_cleaned.groupby('Classificazione')['Vendite_JP'].sum()

# Creazione della figura e degli assi per la griglia 3x2
fig, axs = plt.subplots(3, 2, figsize=(15, 12))

# Vendite per piattaforma
axs[0, 0].bar(vendite_per_piattaforma.index, vendite_per_piattaforma.values, color='skyblue')
axs[0, 0].set_title('Vendite JP per Piattaforma')
axs[0, 0].set_xlabel('Piattaforma')
axs[0, 0].set_ylabel('Vendite JP (Millions)')
axs[0, 0].tick_params(axis='x', rotation=45)
axs[0, 0].grid(True)

# Vendite per genere
axs[0, 1].bar(vendite_per_genere.index, vendite_per_genere.values, color='skyblue')
axs[0, 1].set_title('Vendite JP per Genere')
axs[0, 1].set_xlabel('Genere')
axs[0, 1].set_ylabel('Vendite JP (Millions)')
axs[0, 1].tick_params(axis='x', rotation=45)
axs[0, 1].grid(True)

# Vendite per anno
axs[1, 0].bar(vendite_per_anno.index, vendite_per_anno.values, color='skyblue')
axs[1, 0].set_title('Vendite JP per Anno')
axs[1, 0].set_xlabel('Anno')
axs[1, 0].set_ylabel('Vendite JP (Millions)')
axs[1, 0].tick_params(axis='x', rotation=90)
axs[1, 0].grid(True)

# Vendite per Publisher
axs[1, 1].bar(vendite_per_esrb.index, vendite_per_esrb.values, color='skyblue')
axs[1, 1].set_title('Vendite JP per Voti ESRB')
axs[1, 1].set_xlabel('Voti ESRB')
axs[1, 1].set_ylabel('Vendite JP (Millions)')
axs[1, 1].tick_params(axis='x', rotation=90)
axs[1, 1].grid(True)

# Vendite vs Critic Score
axs[2, 0].scatter(dataset_cleaned['Punteggio_Critici'], dataset_cleaned['Vendite_JP'], color='skyblue', alpha=0.5)
axs[2, 0].set_title('Vendite JP vs Critic Score')
axs[2, 0].set_xlabel('Critic Score')
axs[2, 0].set_ylabel('Vendite JP (Millions)')
axs[2, 0].grid(True)

# Vendite vs User Score
axs[2, 1].scatter(dataset_cleaned['Punteggio_Utenti'], dataset_cleaned['Vendite_JP'], color='skyblue', alpha=0.5)
axs[2, 1].set_title('Vendite JP vs User Score')
axs[2, 1].set_xlabel('User Score')
axs[2, 1].set_ylabel('Vendite JP (Millions)')
axs[2, 1].grid(True)

# Aggiunge layout stretto per migliorare la spaziatura
plt.tight_layout()

# Mostra il grafico
plt.show()



#################################################################################################################
####################################VENDITE ESTERE PER GENERE,###################################################
########################ANNO,CRITIC E USER SCORE, PIATTAFORMA, ESRB RATE#########################################
#################################################################################################################

# Calcola le vendite totali per genere
vendite_per_genere = dataset_cleaned.groupby('Genere')['Altre_Vendite'].sum()

# Calcola le vendite totali per anno
vendite_per_anno = dataset_cleaned.groupby('Anno_di_uscita')['Altre_Vendite'].sum()

# Calcola le vendite totali per piattaforma
vendite_per_piattaforma = dataset_cleaned.groupby('Piattaforma')['Altre_Vendite'].sum()

# Calcola le vendite totali per voti ESBR
vendite_per_esrb = dataset_cleaned.groupby('Classificazione')['Altre_Vendite'].sum()

# Creazione della figura e degli assi per la griglia 3x2
fig, axs = plt.subplots(3, 2, figsize=(15, 12))

# Vendite per piattaforma
axs[0, 0].bar(vendite_per_piattaforma.index, vendite_per_piattaforma.values, color='skyblue')
axs[0, 0].set_title('Vendite Estere per Piattaforma')
axs[0, 0].set_xlabel('Piattaforma')
axs[0, 0].set_ylabel('Vendite Estere (Millions)')
axs[0, 0].tick_params(axis='x', rotation=45)
axs[0, 0].grid(True)

# Vendite per genere
axs[0, 1].bar(vendite_per_genere.index, vendite_per_genere.values, color='skyblue')
axs[0, 1].set_title('Vendite Estere per Genere')
axs[0, 1].set_xlabel('Genere')
axs[0, 1].set_ylabel('Vendite Estere (Millions)')
axs[0, 1].tick_params(axis='x', rotation=45)
axs[0, 1].grid(True)

# Vendite per anno
axs[1, 0].bar(vendite_per_anno.index, vendite_per_anno.values, color='skyblue')
axs[1, 0].set_title('Vendite Estere per Anno')
axs[1, 0].set_xlabel('Anno')
axs[1, 0].set_ylabel('Vendite Estere (Millions)')
axs[1, 0].tick_params(axis='x', rotation=90)
axs[1, 0].grid(True)

# Vendite per Publisher
axs[1, 1].bar(vendite_per_esrb.index, vendite_per_esrb.values, color='skyblue')
axs[1, 1].set_title('Vendite Estere per Voti ESRB')
axs[1, 1].set_xlabel('Voti ESRB')
axs[1, 1].set_ylabel('Vendite Estere (Millions)')
axs[1, 1].tick_params(axis='x', rotation=90)
axs[1, 1].grid(True)

# Vendite vs Critic Score
axs[2, 0].scatter(dataset_cleaned['Punteggio_Critici'], dataset_cleaned['Altre_Vendite'], color='skyblue', alpha=0.5)
axs[2, 0].set_title('Vendite Estere vs Critic Score')
axs[2, 0].set_xlabel('Critic Score')
axs[2, 0].set_ylabel('Vendite Estere (Millions)')
axs[2, 0].grid(True)

# Vendite vs User Score
axs[2, 1].scatter(dataset_cleaned['Punteggio_Utenti'], dataset_cleaned['Altre_Vendite'], color='skyblue', alpha=0.5)
axs[2, 1].set_title('Vendite Estere vs User Score')
axs[2, 1].set_xlabel('User Score')
axs[2, 1].set_ylabel('Vendite Estere (Millions)')
axs[2, 1].grid(True)

# Aggiunge layout stretto per migliorare la spaziatura
plt.tight_layout()

# Mostra il grafico
plt.show()


#################################################################################################################
####################################RELAZIONE TRA VENDITE E AREA#################################################
############################################GEOGRAFICA###########################################################
#################################################################################################################


# Calcola le vendite totali per area geografica
vendite_totali = dataset_cleaned[['Vendite_NA', 'Vendite_EU', 'Vendite_JP', 'Altre_Vendite']].sum()

# Crea un grafico a torta per visualizzare la distribuzione delle vendite
plt.figure(figsize=(8, 8))
plt.pie(vendite_totali, labels=vendite_totali.index, autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'lightcoral', 'lightpink'])
plt.title('Distribuzione delle Vendite per Area Geografica')
plt.show()





#################################################################################################################
####################################REGRESSIONE LINEARE##########################################################
#################################################################################################################
#################################################################################################################

#Regressione lineare tra Critic Score e User score 
# Splitting del dataset in training set (70%), validation set (15%), e test set (15%)
train_ds, rest_ds = train_test_split(dataset_cleaned, test_size=0.3, random_state=42)
val_ds, test_ds = train_test_split(rest_ds, test_size=0.5, random_state=42)

# Variabili predittive e target per la regressione tra Critic_Score e User_Score
X_train_critics = train_ds[['Punteggio_Critici']].values.reshape(-1, 1)
y_train_critics = train_ds['Punteggio_Utenti']

X_val_critics = val_ds[['Punteggio_Critici']].values.reshape(-1, 1)
y_val_critics = val_ds['Punteggio_Utenti']

# Inizializzazione del modello di regressione lineare per Critics vs. Users
model_critics = LinearRegression()
model_critics.fit(X_train_critics, y_train_critics)
y_pred_critics = model_critics.predict(X_val_critics)

# Calcolo dell'errore quadratico medio (MSE) e del coefficiente di determinazione (R^2)
mse_critics = mean_squared_error(y_val_critics, y_pred_critics)
r2_critics = r2_score(y_val_critics, y_pred_critics)
print(f"Errore quadratico medio (MSE) su validation set (Critic vs. User): {mse_critics}")
print(f"Coefficiente di determinazione (R^2) su validation set (Critic vs. User): {r2_critics}")

# Plot dei punti e della regressione tra Critic_Score e User_Score
plt.figure(figsize=(10, 6))
plt.scatter(X_val_critics, y_val_critics, color='blue', label='Valori effettivi')
plt.plot(X_val_critics, y_pred_critics, color='red', linewidth=2, label='Regressione Lineare')
plt.xlabel('Punteggio Critici')
plt.ylabel('Punteggio Utenti')
plt.title('Regressione Lineare tra Punteggio Critici e Punteggio Utenti')
plt.legend()
plt.show()

# Analisi della normalità dei residui
residui_critics = y_val_critics - y_pred_critics
plt.figure(figsize=(8, 5))
plt.hist(residui_critics, bins=30, edgecolor='black')
plt.xlabel('Residui')
plt.ylabel('Frequenza')
plt.title('Distribuzione dei Residui della Regressione (Critics vs. Users)')
plt.show()

# QQ plot per i residui (Critic vs. User)
plt.figure(figsize=(8, 5))
stats.probplot(residui_critics, dist="norm", plot=plt)
plt.title('QQ Plot dei Residui (Critics vs. Users)')
plt.show()

# Test di Shapiro-Wilk per i residui (Critic vs. User)
shapiro_test_critics = shapiro(residui_critics)
print(f"Test di Shapiro-Wilk (Critics vs. Users): Statistica = {shapiro_test_critics[0]}, p-value = {shapiro_test_critics[1]}\n\n")



# Regressione lineare tra Punteggio Utenti e anno di uscita
# Variabili predittive e target per la regressione tra Year_of_Release e User_Score
X_train_year = train_ds[['Anno_di_uscita']].values.reshape(-1, 1)
y_train_year = train_ds['Punteggio_Utenti']

X_val_year = val_ds[['Anno_di_uscita']].values.reshape(-1, 1)
y_val_year = val_ds['Punteggio_Utenti']

# Inizializzazione del modello di regressione lineare per Year_of_Release vs. User_Score
model_year = LinearRegression()
model_year.fit(X_train_year, y_train_year)
y_pred_year = model_year.predict(X_val_year)

# Calcolo dell'errore quadratico medio (MSE) e del coefficiente di determinazione (R^2)
mse_year = mean_squared_error(y_val_year, y_pred_year)
r2_year = r2_score(y_val_year, y_pred_year)
print(f"Errore quadratico medio (MSE) su validation set (Year vs. User): {mse_year}")
print(f"Coefficiente di determinazione (R^2) su validation set (Year vs. User): {r2_year}")

# Plot dei punti e della regressione tra Year_of_Release e User_Score
plt.figure(figsize=(10, 6))
plt.scatter(X_val_year, y_val_year, color='blue', label='Valori effettivi')
plt.plot(X_val_year, y_pred_year, color='red', linewidth=2, label='Regressione Lineare')
plt.xlabel('Anno di Uscita')
plt.ylabel('Punteggio Utenti')
plt.title('Regressione Lineare tra Anno di Uscita e Punteggio Utenti')
plt.legend()
plt.show()

# Analisi della normalità dei residui
residui_year = y_val_year - y_pred_year
plt.figure(figsize=(8, 5))
plt.hist(residui_year, bins=30, edgecolor='black')
plt.xlabel('Residui')
plt.ylabel('Frequenza')
plt.title('Distribuzione dei Residui della Regressione (Year vs. Users)')
plt.show()

# QQ plot per i residui (Year vs. User)
plt.figure(figsize=(8, 5))
stats.probplot(residui_year, dist="norm", plot=plt)
plt.title('QQ Plot dei Residui (Year vs. Users)')
plt.show()

# Test di Shapiro-Wilk per i residui (Year vs. User)
shapiro_test_year = shapiro(residui_year)
print(f"Test di Shapiro-Wilk (Year vs. Users): Statistica = {shapiro_test_year[0]}, p-value = {shapiro_test_year[1]}\n\n")



#################################################################################################################
####################################ADDESTRAMENTO MODELLO########################################################
#################################################################################################################
#################################################################################################################


# Trasformo variabile target in variabile categorica
intervalli = [0, 0.2, 3, float('inf')]
categorie = ['poco', 'medio', 'tanto']
dataset_cleaned["Vendite_Globali"] = pd.cut(dataset_cleaned["Vendite_Globali"], bins=intervalli, labels=categorie)

# Codifica delle variabili categoriali
dataset_encoded = pd.get_dummies(dataset_cleaned, columns=['Piattaforma', 'Genere', 'Classificazione', 'Anno_di_uscita'])

# Suddivisione del dataset in training set (70%), validation set (15%) e test set (15%)
train_set, rest_set = train_test_split(dataset_encoded, test_size=0.3, random_state=42)
val_set, test_set = train_test_split(rest_set, test_size=0.5, random_state=42)

# Separazione delle variabili di input e della variabile target per training set
X_train = train_set.drop(['Vendite_Globali'], axis=1)
y_train = train_set['Vendite_Globali']

# Separazione delle variabili di input e della variabile target per validation set
X_val = val_set.drop(['Vendite_Globali'], axis=1)
y_val = val_set['Vendite_Globali']

# Separazione delle variabili di input e della variabile target per test set
X_test = test_set.drop(['Vendite_Globali'], axis=1)
y_test = test_set['Vendite_Globali']

# Standardizzazione delle variabili di input
scaler = StandardScaler()

# Adatta lo scaler sui dati di training e trasforma i dati di training
X_train_scaled = scaler.fit_transform(X_train)

# Trasforma i dati di validazione e test utilizzando lo scaler addestrato sui dati di training
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Addestramento del modello SVM con kernel lineare
model_svm = SVC(kernel='linear', probability=True)
model_svm.fit(X_train_scaled, y_train)

# Predizione sul validation set
y_pred_svm_val = model_svm.predict(X_val_scaled)

# Calcolo delle metriche
accuracy_svm_val = accuracy_score(y_val, y_pred_svm_val)
conf_matrix_svm_val = confusion_matrix(y_val, y_pred_svm_val)
class_report_svm_val = classification_report(y_val, y_pred_svm_val)

print(f"Accuracy (Validation Set): {accuracy_svm_val}\n\n")


# Visualizzazione della Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_svm_val, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predetto')
plt.ylabel('Effettivo')
plt.title('Matrice di Confusione (Validation Set) SVC linear ')
plt.show()

#################################################################################################################
####################################HYPERPARAMETER TUNING########################################################
#################################################################################################################
#################################################################################################################


# Hyperparameter Tuning per SVM
migliore_accuracy = 0
migliore_grado = 0

print("\nHyperparameter Tuning per il modello SVM")
for grado in range(1, 8):
    modello = SVC(kernel="poly", degree=grado)
    modello.fit(X_train_scaled, y_train)
    y_pred_val = modello.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, y_pred_val)
    print(f"Accuratezza per SVM con grado {grado}: {accuracy:.4f}")
    if accuracy > migliore_accuracy:
        migliore_accuracy = accuracy
        migliore_grado = grado

print(f"Miglior grado per SVM: {migliore_grado}")

# Addestramento del modello SVM con il miglior grado
migliore_svm = SVC(kernel="poly", degree=migliore_grado, probability=True)
migliore_svm.fit(X_train_scaled, y_train)

# Predizione sul validation set
y_pred_svm_val = migliore_svm.predict(X_val_scaled)

# Calcolo delle metriche
accuracy_svm_val = accuracy_score(y_val, y_pred_svm_val)
conf_matrix_svm_val = confusion_matrix(y_val, y_pred_svm_val)
class_report_svm_val = classification_report(y_val, y_pred_svm_val)

print(f"Accuratezza (Validation Set): {accuracy_svm_val}\n\n")


# Visualizzazione della Matrice di Confusione
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_svm_val, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predetto')
plt.ylabel('Effettivo')
plt.title('Matrice di Confusione (Validation Set) SVC poly')
plt.show()


#################################################################################################################
####################################STUDIO STATISTICO############################################################
######################################DEI RISULTATI##############################################################
#################################################################################################################


# Numero di ripetizioni per l'esecuzione del modello
k = 10

# Liste per memorizzare le performance
accuratezze = []
matrici_di_confusione = []

# Esecuzione del modello per k volte
for _ in range(k):
    train_set, rest_set = train_test_split(dataset_encoded, test_size=0.3, random_state=None)
    val_set, test_set = train_test_split(rest_set, test_size=0.5, random_state=None)

    X_train = train_set.drop(['Vendite_Globali'], axis=1)
    y_train = train_set['Vendite_Globali']
    X_val = val_set.drop(['Vendite_Globali'], axis=1)
    y_val = val_set['Vendite_Globali']
    X_test = test_set.drop(['Vendite_Globali'], axis=1)
    y_test = test_set['Vendite_Globali']

    # Standardizzazione delle variabili di input
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Addestramento del modello
    modello = SVC(kernel="linear",probability=True)
    modello.fit(X_train_scaled, y_train)

    # Valutazione del modello sul test set
    y_pred_test = modello.predict(X_test_scaled)
    accur = accuracy_score(y_test, y_pred_test)
    conf_matrix = confusion_matrix(y_test, y_pred_test)

    # Memorizzazione delle metriche
    accuratezze.append(accur)
    matrici_di_confusione.append(conf_matrix)

# Calcolo del centro e della diffusione dei dati
media_accuratezze = np.mean(accuratezze)
std_accuratezze = np.std(accuratezze)

print(f"Media dell'Accuratezza su {k} esecuzioni: {media_accuratezze:.4f}")
print(f"Deviazione Standard dell'Accuratezza su {k} esecuzioni: {std_accuratezze:.4f}")

# Istogramma dell'Accuratezza
plt.figure(figsize=(8, 6))
plt.hist(accuratezze, bins=10, edgecolor='black')
plt.xlabel('Accuratezza')
plt.ylabel('Frequenza')
plt.title(f'Distribuzione dell\'Accuratezza su {k} esecuzioni')
plt.show()

# Boxplot dell'Accuratezza
plt.figure(figsize=(8, 6))
plt.boxplot(accuratezze, vert=False, patch_artist=True, widths=0.7)
plt.xlabel('Accuratezza')
plt.title(f'Boxplot dell\'Accuratezza su {k} esecuzioni')
plt.show()

# Calcolo dell'intervallo di confidenza per l'accuratezza

alpha = 0.05  
df = k - 1  
t_score = t.ppf(1 - alpha / 2, df)

stderr_accuratezze = std_accuratezze / np.sqrt(k)  
intervallo_confidenza = t_score * stderr_accuratezze  

media_accuratezze, intervallo_confidenza_low, intervallo_confidenza_up = \
    np.mean(accuratezze), media_accuratezze - intervallo_confidenza, media_accuratezze + intervallo_confidenza

print(f"Intervallo di Confidenza del 95% per l'Accuratezza: [{intervallo_confidenza_low:.4f}, {intervallo_confidenza_up:.4f}]")




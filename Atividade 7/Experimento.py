import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN


data = pd.read_csv('dados.csv')
x = data.iloc[:, 1:]

modelo_ia_1 = KMeans(n_clusters=5)
modelo_ia_2 = HDBSCAN(min_cluster_size=2)
modelo_ia_1.fit(x)
modelo_ia_2.fit(x)

data['Grupo KMeans'] = modelo_ia_1.labels_
data['Grupo HDBSCAN'] = modelo_ia_2.labels_
data = data.sort_values(by=['Grupo KMeans'])
data.to_csv('resultado.csv', index=False)
print(data)


centros = pd.DataFrame(modelo_ia_1.cluster_centers_, columns=data.columns.values[1:-2])
centros = centros.round(2)
centros['Nome'] = ['Centro ' + str(i) for i in range(modelo_ia_1.n_clusters)]
centros.to_csv('grupos.csv', index=False)
print(centros)

import os #libreria per la gestione percorsi
import pandas as pd #libreria per la gestione delle tabelle e per la manipolazione di dati strutturati come excel e CSV
import SimpleITK as sitk #libreria per la gestione delle immagini, segmentazione, registrazione e filtraggio delle immagini mediche .mha
import numpy as np #libreria per array
import matplotlib.pyplot as plt #libreria per plottare l'immagine

#dobbiamo definire le variabili che conterranno i path delle immagini d'interesse
#path = r"C:\Users\aless\OneDrive - Università Campus Bio-Medico di Roma\Attachments\Campus_Biomedico\Dottorato\Lavoro\Prog_Dott\Studio_GAN\Dataset\synthRAD2025\synthRAD2025_Task1_Train\Task1\AB"
path = r"C:\Users\aless\Documents\Dataset\synthRAD2025\synthRAD2025_Task1_Train\Task1\AB"#r mi evita problemi con \
filepath= os.path.join(path, "overviews", "1_AB_train_parameters.xlsx") #aggiunge al percorso definito la cartella d'interesse e il file excel, praticamente aggiunge parti del percorso
#perchè non ho potuto mettere tutto nel primo percorso? già tutto in Path?
data = pd.read_excel(filepath) #mi permette di leggere la tabella excel che sta in quel percorso. In questo modo definisco la tabella "data"
print(data)
print(data.loc[0,"ID"]) #ricordati che questa funzione mi rida il valore di questa cella che in questo caso è una stringa e la funzione di seguito
#os.path.join(path, "", "") prende solo stringhe.
filepath_1= os.path.join(path, data.loc[0,"ID"], "ct.mha") #sostanzialmente qui mi indica il percorso che mi interessa per quel tipo d'immagine
#voglio solo l'immagine TC(ct.mha) che ha l'indice "1ABA005" riportato nella tabella e che sta nella cartella 1ABA005
#print(os.path.exists(path))
image = sitk.ReadImage(filepath_1) #il file ct.mha si carica con questa funzione, così riesco a caricare la specifica immagine e a leggerla
# Converti in array NumPy se necessario
image_array = sitk.GetArrayFromImage(image) #Converte l'immagine in un array NumPy 3D con formato [z,y,x] se stiamo parlando di 3 dimensioni(Slice, Height, width).
print(image_array.shape[0])
central_slice_index = image_array.shape[0] // 2 #prende l'indice centrale delle slice lungo l'asse z (il totale diviso per 2)
plt.imshow(image_array[central_slice_index, :, :], cmap='gray') #visualizza solo un immagine in formato 2D dicendo di prendere la slice numero 45 e prendere tutti
#i pixel riga x colonna in scala di grigi
plt.title(f"Slice che mi interessa (indice {central_slice_index})") #questa funzione mi permette di aggiungere un titolo alla slice dove la parte in verde è cio
#che scrivo io e dentro le parentesi ho la variabile dell'indice desiderato
plt.axis('off') #rimuove i mumeri e i bordi dagli assi del grafico per rendere l'immagine più pulita
plt.show() #serve per mostrare l'immagine
print(image_array.shape) #leggo le dimensioni dell'array, anche se non coincidono
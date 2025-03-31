# Descriere proiect

## preprocess.py
Este un script folosit o singura data pentru a crea fisierele .csv din directorul csv_files.

## dataset.py
Defineste clasa de Dataset si functia collate_fn care rezolva inconsistenta studiilor cu numar variabil de imagini.

## train.py
Defineste modelul folosit si functiile de antrenare/evaluare. In cazul baseline-ului (care este copiat din studiul MURA atasat), se foloseste:
- Backbone: Densenet169 preantrenat pe ImageNet
- Augmentari: Inversii laterale si rotatii de maxim 30 grade
- Batch Size: 8
- Metoda de predictie: Prob Mean (fiecare imagine dintr-un studiu genereaza o probabilitate, iar apoi media probabilitatilor imaginilor dintr-un studiu reprezinta predictia finala)
- Loss: WBCE (implementat cu Focal Loss)
- Optimizator: Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
- Scheduler: ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

## main.py
Defineste dataseturile, dataloaderele, transformarile si hiperparametrii antrenamentului. Ruleaza antrenamentul si evalueaza modelul.

## plot.py
Contine functii pentru plotatul rezultatelor.



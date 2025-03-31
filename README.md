# Descriere proiect

## preprocess.py
Este un script folosit o singura data pentru a crea fisierele .csv din directorul csv_files.

## dataset.py
Defineste clasa de Dataset si functia collate_fn care rezolva inconsistenta studiilor cu numar variabil de imagini.

## train.py
Defineste modelul folosit si functiile de antrenare/evaluare. In cazul baseline-ului (care este copiat din studiul MURA atasat), se foloseste:
- Backbone-ul Densenet169 preantrenat pe ImageNet
- Metoda de predictie: Prob Mean (fiecare imagine dintr-un studiu genereaza o probabilitate, iar apoi media probabilitatilor imaginilor dintr-un studiu reprezinta predictia finala)
- Loss: WBCE (implementat cu Focal Loss)

## main.py
Defineste dataseturile, dataloaderele, transformatile si hiperparametrii antrenamentului. Ruleaza antrenamentul si evalueaza modelul.

## plot.py
Contine functii pentru plotatul rezultatelor.



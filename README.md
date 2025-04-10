# Descriere proiect

## train.py
Defineste modelul folosit si functiile de antrenare/evaluare. In cazul baseline-ului configuratia este copiata din studiul MURA atasat (sectiunea 3) si presupune:
- Backbone: Densenet169 preantrenat pe ImageNet
- Augmentari: Inversari laterale si rotatii de maxim 30 grade si redimensionare la 320x320
- Batch Size: 8
- Metoda de predictie: Prob Mean (fiecare imagine dintr-un studiu genereaza o probabilitate, iar apoi media probabilitatilor imaginilor dintr-un studiu reprezinta predictia finala)
- Loss: WBCE 
- Optimizator: Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
- Scheduler: ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

  Pentru fiecare batch, se adauga imagini de padding studiilor cu numar mai mic de imagini astfel incat toate sa aiba aceeasi dimensiune si sa poata fi procesate in parelel.
  La final, predictia este agregata cu ajtorul unei masti de padding (daca studiul contine 3 imagini originale si 2 de padding, masca este [1,1,1,0,0]).

## main.py
Defineste dataseturile, dataloaderele, transformarile si hiperparametrii antrenamentului. Ruleaza antrenamentul si evalueaza modelul.

## plot.py
Contine functii pentru plotatul rezultatelor.

## preprocess.py
Este un script folosit o singura data pentru a crea fisierele .csv din directorul csv_files.

## dataset.py
Defineste clasa de Dataset si functia collate_fn care rezolva inconsistenta studiilor cu numar variabil de imagini.

## Attention Mechanism

1. Fiecare imagine este procesată individual printr-un backbone CNN DenseNet169 pentru a extrage un vector de caracteristici (dimensiune 512).
2. Pentru a procesa eficient pe GPU, studiile sunt padded astfel incat toate sa aiba acelasi numar de imagini si acompaniate de o masca care marcheaza care imagini sunt reale vs. padding.
3. Fiecare vector de caracteristici (al fiecarei imagini) este trecut printr-un clasificator simplu MLP pentru a obtine un scor (logit) per imagine.
4. Se calculeaza o medie ponderata a vectorilor de caracteristici folosind scorurile de atentie. Rezultatul este un context vector pentru fiecare studiu: 512.
5. Contextul este trecut printr-un classifier MLP. Se obține un singur logit per studiu (predicția modelului).
6. (Optional) Loss-ul total este calculat ca si loss-ul contextului final si loss-ul fiecarei imagini: loss = study_loss + 0.3 * image_loss



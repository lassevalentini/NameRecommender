# NameRecommender

Developed using the [Anaconda distribution](https://www.anaconda.com/distribution/).

A list of legal danish names can be found [here](https://ast.dk/born-familie/hvad-handler-din-klage-om/navne/navnelister/godkendte-fornavne). 
It needs to be exported to a simple text file with only one name per line.

## Running

Create and activate a new conda env with:

```bash
conda create -n name_recommender -f requirements.txt
conda activate name_recommender
``` 

Run and train the model with 
```bash
python name_recommender.py navne.txt
```

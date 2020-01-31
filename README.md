# Machine learning meets p*K*<sub>a</sub>

## Prerequisites

The Python dependencies are:
* Python >= 3.7
* NumPy >= 1.18
* Scikit-Learn >= 0.22
* RDKit >= 2019.09.3
* Pandas >= 0.25
* XGBoost >= 0.90
* JupyterLab >= 1.2
* Matplotlib >= 3.1
* Seaborn >= 0.9

For the data preparation pipeline, *ChemAxon Marvin*<sup>[1]</sup> and 
*OpenEye QUACPAC/Tautomers*<sup>[2]</sup> are required. To use the provided 
prediction model with the included Python script, *ChemAxon Marvin*<sup>[1]</sup> 
is <ins>not</ins> required.

Of course you also need the code from this repository folder.

### Installing

First of all you need a working Miniconda/Anaconda installation. You can get
Miniconda at https://conda.io/en/latest/miniconda.html.

Now you can create an environment named "mmws_2020" with all needed dependencies and
activate it with:
```bash
conda env create -f environment.yml
conda activate mmws_2020
```

You can also create a new environment by yourself and install all dependencies without the
environment.yml file:
```bash
conda create -n my_env python=3.7
conda activate my_env
```
In case of Linux or macOS:
````bash
conda install -c defaults -c rdkit -c conda-forge scikit-learn rdkit xgboost jupyterlab matplotlib seaborn
````

In case of Windows:
```bash
conda install -c defaults -c rdkit scikit-learn rdkit jupyterlab matplotlib seaborn
pip install xgboost
```

## Usage
### <a name="prep"></a>Preparation pipeline
To use the data preparation pipeline you have to be in the repository folder and your conda
environment have to be activated. Additionally the *Marvin* commandline tool `cxcalc` and
the *QUACPAC* commandline tool `tautomers` have to be contained in your `PATH` variable.

Also the environment variables `OE_LICENSE` (containing the path to your *OpenEye* license
file) and `JAVA_HOME` (referring to the *Java* installation folder, which is needed for 
`cxcalc`) have to be set.

After preparation you can display a small usage information with `bash run_pipeline.sh -h`.
Example call:
```bash
bash run_pipeline.sh --train datasets/chembl25.sdf --test datasets/novartis_cleaned_mono_unique_notraindata.sdf
```

### Prediction tool
First of all you have to be in the repository folder and your conda environment have
to be activated. To use the prediction tool you have to retrain the machine learning model.
Therefore just call the training script, it will train the 5-fold cross-validated Random
Forest machine learning model <ins>using **12** cpu cores</ins>. If you want to adjust the number of 
cores you can edit the train_model.py by changing the value of the variable `EST_JOBS`.
```bash
python train_model.sdf
```
To use the prediction tool with the trained model *QUACPAC/Tautomers* have to be available 
as it was mentioned in the [chapter above](#prep).

Now you can call the python script with a SDF file and an output path:
```bash
python predict_sdf.py my_test_file.sdf my_output_file.sdf
```

**NOTE:** This model was build for monoprotic structures regarding a pH range of 2 to 12.
If the model is used with multiprotic structures, the predicted values will probably not
be correct.

## Datasets

1. `AvLiLuMoVe.sdf` - Manually combined literature p<i>K</i><sub>a</sub> data<sup>[3]</sup>
2. `chembl25.sdf` - Experimental p<i>K</i><sub>a</sub> data extracted from ChEMBL25<sup>[4]</sup>
3. `datawarrior.sdf` - p<i>K</i><sub>a</sub> data shipped with DataWarrior<sup>[5]</sup>
4. `combined_training_datasets_unique.sdf` -  [Preprocessed](#prep) and combined data 
from datasets (2) and (3), used as training dataset
5. `AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf` - [Preprocessed](#prep) data from dataset (1),
used as external testset
6. `novartis_cleaned_mono_unique_notraindata.sdf` - [Preprocessed](#prep) data from an inhouse
dataset provided by Novartis<sup>[6]</sup>, used as external testset

## Authors

**Marcel Baltruschat** - [GitHub](https://github.com/mrcblt), [E-Mail](mailto:marcel.baltruschat@tu-dortmund.de)<br>
**Paul Czodrowski** - [GitHub](https://github.com/czodrowskilab), [E-Mail](mailto:paul.czodrowski@tu-dortmund.de)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## References

[1] *Marvin* 20.1.0, 2020, ChemAxon, [http://www.chemaxon.com](http://www.chemaxon.com)<br>
[2] *QUACPAC* 2.0.2.2: OpenEye Scientific Software, Santa Fe, NM. [http://www.eyesopen.com](http://www.eyesopen.com)<br>
[3] Settimo, L., Bellman, K. & Knegtel, R.M.A. Pharm Res (2014) 31: 1082. 
[https://doi.org/10.1007/s11095-013-1232-z](https://doi.org/10.1007/s11095-013-1232-z)<br>
[4] Gaulton A, Hersey A, Nowotka M, Bento AP, Chambers J, Mendez D, Mutowo P, Atkinson F, 
Bellis LJ, Cibrián-Uhalte E, Davies M, Dedman N, Karlsson A, Magariños MP, Overington JP, 
Papadatos G, Smit I, Leach AR. (2017) 'The ChEMBL database in 2017.' Nucleic Acids Res., 
45(D1) D945-D954.<br>
[5] Thomas Sander, Joel Freyss, Modest von Korff, Christian Rufener. DataWarrior: An Open-Source 
Program For Chemistry Aware Data Visualization And Analysis. J Chem Inf Model 
2015, 55, 460-473, doi 10.1021/ci500588j<br>
[6] Richard A. Lewis, Stephane Rodde, Novartis Pharma AG, Basel, Switzerland
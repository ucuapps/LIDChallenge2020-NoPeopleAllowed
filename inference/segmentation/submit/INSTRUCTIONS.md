# UCU & SoftServe team - Inference instructions

a. Create a virtual Python3.6+ environment and activate it
```
python3 -m venv inference_env
source inference_env/bin/activate
```

b. Install the requirements
```
cd submit
pip install -r requirements_inference.txt
```

c. Run inference. Select the path where to save predicted masks and provide path to the data folder (either validation, or test).
The script automatically uses `'cuda:0'` device. To use a different device, pass this parameter: `--device 'cuda:[SELECT_ANY]'`. 

```
python generate_predictions.py [SAVE_PATH] [DATA_PATH] 
# Example: python generate_predictions.py ~/results_val ~/LID_track1/val/ --device 'cuda:1'
```

Predictions are stored in the folder, that was specified as `SAVE_PATH`. 
It can be zipped to `results.zip` and used for submission.


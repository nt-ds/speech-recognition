# speech-recognition

### Directory structure
--- main folders and files only ---

		speech-recognition
			data (input data)
				partitioned (generated)
					test
						30 folders with wav files
					training
						30 folders with wav files
					validation
						30 folders with wav files
						
				processed (generated)
					oscillograms
						test
							30 folders with png files
						training
							30 folders with png files
						validation
							30 folders with png files
							
					spectrograms
						test
							30 folders with png files
						training
							30 folders with png files
						validation
							30 folders with png files
							
				raw (obtained from [Kaggle](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge))
					test
						'audio' folder with wav files
						
					train
						audio
							31 folders with wav files
						README.md
						testing_list.txt
						validation_list.txt
			
			models (saved models with weights and their performances)
				oscillograms
					6 hdf5 files and 6 png files
				
				spectrograms
					18 hdf5 files and 18 png files
						
			notebooks (executable code)
			
			result (evaluations of the best model)
			
			src (source code)
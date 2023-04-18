from flask import * 
import pickle
from transformers import pipeline , AutoModelForSeq2SeqLM, AutoTokenizer
import torch

#Initialize the flask App
app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_ckpt = "google/pegasus-cnn_dailymail"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

pipe = pipeline('summarization', model = model_ckpt)
pickle.dump(pipe, open('model.pkl','wb'))
model = pickle.load(open('model.pkl', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template("index.html")

#To use the predict button in our web-app
@app.route('/summarize',methods=['POST'])
def summarize():
   

    
    if request.method == 'POST':
        if request.form['inputText'] : 
            text = request.form['inputText'] 
            print(text)
            #For rendering results on HTML GUI
            gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 90}
            model_sum = model(text, **gen_kwargs)[0]["summary_text"]
            print(model_sum)
            return render_template('index.html', summarized_text=' {}'.format(model_sum))  
        else :  
            return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True)
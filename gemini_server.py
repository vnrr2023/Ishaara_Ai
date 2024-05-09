import google.generativeai as palm
palm.configure(api_key="<Put Your Api Key Here>")
from flask import Flask,jsonify,request
from flask_cors import CORS
app = Flask(__name__)
model = palm.GenerativeModel('gemini-pro')
CORS(app)

@app.route('/gemini_api', methods=['POST'])
def gemini_api():
    text = request.form.get('text', '')
    if text == "":
        return {'data': 'please pass data'}
    prompt = f'''{text} 
    make a meaningfull sentence from these words (is,are,was,were,am etc)
    if grammar is there dont change it and also dont change the original words. dont give asteriks and \n and use proper grammar
    '''
    response = model.generate_content(prompt)
    print(response.text)
    return jsonify({
        'data': str(response.text)
    })

if __name__ == '__main__':
    app.run(debug=True)
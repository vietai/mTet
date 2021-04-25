import streamlit as st
import streamlit.components.v1 as components
import SessionState
import time 

import os
import text_encoder
import six
import tensorflow as tf
import base64
import numpy as np
import copy
import re
from PIL import Image

import googleapiclient.discovery
from google.api_core.client_options import ClientOptions

from google.cloud import firestore
gg_api_credential = 'vietai-research-mlmodel.json'
if os.path.exists(gg_api_credential):
  os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gg_api_credential # change for your GCP key
else:
  os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ''

PROJECT = "vietai-research" # change for your GCP project
REGION = "asia-southeast1" # change for your GCP region (where your model is hosted)
MODEL = "translation_appendtag_envi_base_1000k"
ENVI_VERSION = 'envi_pure_tall9'
vocab_file = 'vocab.subwords'


def to_example(dictionary):
  """Helper: build tf.Example from (string -> int/float/str list) dictionary."""
  features = {}
  for (k, v) in six.iteritems(dictionary):
    if not v:
      raise ValueError("Empty generated field: %s" % str((k, v)))
    # Subtly in PY2 vs PY3, map is not scriptable in py3. As a result,
    # map objects will fail with TypeError, unless converted to a list.
    if six.PY3 and isinstance(v, map):
      v = list(v)
    if (isinstance(v[0], six.integer_types) or
        np.issubdtype(type(v[0]), np.integer)):
      features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    elif isinstance(v[0], float):
      features[k] = tf.train.Feature(float_list=tf.train.FloatList(value=v))
    elif isinstance(v[0], six.string_types):
      if not six.PY2:  # Convert in python 3.
        v = [bytes(x, "utf-8") for x in v]
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    elif isinstance(v[0], bytes):
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    else:
      raise ValueError("Value for %s is not a recognized type; v: %s type: %s" %
                       (k, str(v[0]), str(type(v[0]))))
  example = tf.train.Example(features=tf.train.Features(feature=features))
  return example.SerializeToString()


def get_resource(version):
    if os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == '':
      return None, None
    # Create the ML Engine service object
    prefix = "{}-ml".format(REGION) if REGION else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)

    # Setup model path
    model_path = "projects/{}/models/{}".format(PROJECT, MODEL)
    if version is not None:
        model_path += "/versions/{}".format(version)

    # Create ML engine resource endpoint and input data
    predictor = googleapiclient.discovery.build(
        "ml", "v1", cache_discovery=False, client_options=client_options).projects()
    return predictor, model_path


def check_mrs(content, i):
  is_mr = (i >= 2 and 
           content[i-2:i].lower() in ['mr', 'ms'] and
           (i < 3 or content[i-3] == ' '))
  is_mrs = (i >= 3 and 
            content[i-3:i].lower() == 'mrs' and 
            (i < 4 or content[i-4] == ' '))
  return is_mr or is_mrs


def check_ABB_mid(content, i):
  if i <= 0:
    return False
  if i >= len(content)-1:
    return False
  l, r = content[i-1], content[i+1]
  return l.isupper() and r.isupper()


def check_ABB_end(content, i):
  if i <= 0:
    return False
  l = content[i-1]
  return l.isupper()


def normalize(contents):
  # first step: replace special characters 
  check_list = ['\uFE16', '\uFE15', '\u0027','\u2018', '\u2019',
                '“', '”', '\u3164', '\u1160', 
                '\u0022', '\u201c', '\u201d', '"',
                '[', '\ufe47', '(', '\u208d',
                ']', '\ufe48', ')' , '\u208e', 
                '—', '_', '–', '&']
  alter_chars = ['?', '!', '&apos;', '&apos;', '&apos;',
                 '&quot;', '&quot;', '&quot;', '&quot;', 
                 '&quot;', '&quot;', '&quot;', '&quot;', 
                 '&#91;', '&#91;', '&#91;', '&#91;',
                 '&#93;', '&#93;', '&#93;', '&#93;', 
                 '-', '-', '-', '&amp;']

  replace_dict = dict(zip(check_list, alter_chars))

  new_contents = ''
  for i, char in enumerate(contents):
    if char == '&' and (contents[i:i+5] == '&amp;' or
                        contents[i:i+6] == '&quot;' or
                        contents[i:i+6] == '&apos;' or
                        contents[i:i+5] == '&#93;' or
                        contents[i:i+5] == '&#91;'):
      new_contents += char
      continue
    new_contents += replace_dict.get(char, char)
  contents = new_contents

  # second: add spaces
  check_sp_list = [',', '?', '!', '&apos;', '&amp;', '&quot;', '&#91;', 
                   '&#93;', '-', '/', '%', ':', '$', '#', '&', '*', ';', '=', '+', '@', '~', '>', '<']

  new_contents = ''
  i = 0
  while i < len(contents):
    char = contents[i]
    found = False
    for string in check_sp_list:
      if string == contents[i: i+len(string)]:
        new_contents += ' ' + string 
        if string != '&apos;':
          new_contents += ' '
        i += len(string)
        found = True
        break
    if not found:
      new_contents += char
      i += 1
  contents = new_contents

  # contents = contents.replace('.', ' . ')
  new_contents = ''
  for i, char in enumerate(contents):
    if char != '.':
      new_contents += char
      continue    
    elif check_mrs(contents, i):
      # case 1: Mr. Mrs. Ms.
      new_contents += '. '
    elif check_ABB_mid(contents, i):
      # case 2: U[.]S.A.
      new_contents += '.'
    elif check_ABB_end(contents, i):
      # case 3: U.S.A[.]
      new_contents += '. '
    else:
      new_contents += ' . '

  contents = new_contents
  
  # third: remove not necessary spaces.
  new_contents = ''
  for char in contents:
    if new_contents and new_contents[-1] == ' ' and char == ' ':
      continue
    new_contents += char
  contents = new_contents
  
  return contents.strip()


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


@st.cache
def translate(from_txt, direction):
  # we need the argument direction here
  # so that translate(...) will not cache input
  # from envi to vien and vice versa.
  from_txt = from_txt.strip()
  input_ids = state.encoder.encode(from_txt) + [1]
  input_ids += [0] * (128 - len(input_ids))
  byte_string = to_example({
      'inputs': list(np.array(input_ids, dtype=np.int32))
  })
  content = base64.b64encode(byte_string).decode('utf-8')

  # and the json to send is:
  input_data_json = {
      "signature_name": "serving_default",
      "instances": [{"b64": content}]
  } 

  request = state.model.predict(name=state.model_path, body=input_data_json)
  response = request.execute()
  
  if "error" in response:
      raise RuntimeError(response["error"])

  translated_text = ''
  for idx in response['predictions'][0]['outputs']:
      translated_text += state.vocab[idx][1:-1]
    
  to_text = translated_text.replace('_', ' ')
  to_text = re.sub('\s+', ' ', to_text)

  to_text = to_text.replace('<EOS>', '').replace('<pad>', '')
  to_text = to_text.replace('& quot ;', '"')
  to_text = to_text.replace(' & apos ;', "'")
  to_text = to_text.replace('&# 91 ; ', "(")
  to_text = to_text.replace(' &# 93 ;', ")")
  to_text = to_text.replace(' , ', ', ')
  to_text = to_text.replace(' . ', ". ")
  to_text = to_text.replace(' : ', ": ")
  to_text = to_text.split('\\')[0].strip()

  return to_text


def join_each_https(from_to):
  ref, translate = from_to
  split_at = './#?!=-_: '
  
  while translate:
    idx = min([translate.index(c) if c in translate else 1000 
               for c in split_at])
    t = translate[:idx]
    if t == '':
      pass
    elif t not in ref:
      break
    translate = translate[idx+1:]
  
  return ref + ' ' + translate


def join_multiple_https(from_txt, to_txt):
  from_txt = from_txt.split(' ')
  ref_https = [t for t in from_txt if t.startswith('https://')]
  
  to_txt = to_txt.split('https: / /')
  translate_https = to_txt[1:]
  if len(ref_https) < len(translate_https):
    ref_https += [''] * (len(translate_https) - len(ref_https))
  else:
    ref_https = ref_https[:len(translate_https)]
  
  translate_https = map(join_each_https, zip(ref_https, translate_https))
  return ''.join(to_txt[:1] + list(translate_https))


def write_ui():
  state.from_txt = st.text_area('Enter text to translate and click "Translate" ',
                            value=state.prompt,
                            height=100,
                            max_chars=600)
  
  button_value = st.button('Translate')
  state.ph0 = st.empty()

  if state.first_time :
    state.text_to_show = ''
  else:
    state.from_txt = state.from_txt.replace('\n', ' ')
    normalized = normalize(state.from_txt)
    if state.model is None or state.model_path is None:
      state.text_to_show = 'Google Cloud Platform access is currently disabled. This is a dummy translation.'
    else:  
      state.text_to_show = translate(
          normalized, state.direction_choice)
  
  state.text_to_show = join_multiple_https(
      state.from_txt,
      state.text_to_show
  )

  state.first_time = False
  
  state.user_edit = state.ph0.text_area(
                    'Translated text',
                    height=100,
                    value=state.text_to_show)

  if button_value or state.like:
    state.like = True
  
  different = normalize(state.user_edit) != normalize(state.text_to_show)
  
  state.col1, state.col2, state.col3, state.col4 = st.beta_columns([0.9, 0.25, 0.25, 3.0])
  with state.col2:
    state.ph2 = st.empty()

  if state.like:
    with state.col2:
      state.ph2 = st.empty()
      state.b2 = state.ph2.button('Yes')  
    with state.col3:
      state.ph3 = st.empty()
      state.b3 = state.ph3.button('No')   
    with state.col1:
      state.ph1 = st.markdown('Is this translation good ?')
  
    if state.b2:
      state.like = False
      state.ph1.empty()
      state.ph2.empty()
      state.ph3.empty()
      if state.db is None:
        st.success('Google Cloud Platform access is currently disabled. Suggestions will be recorded once access is enabled.')
      else:
        if state.direction_choice == "English to Vietnamese":
          state.db.collection(u"envi").add({
              u'from_text': state.from_txt,
              u'model_output': state.text_to_show,
              u'user_approve': True,
              u'time': time.time()
          })
        else:
          state.db.collection(u"vien").add({
              u'from_text': state.from_txt,
              u'model_output': state.text_to_show,
              u'user_approve': True,
              u'time': time.time()
          })
        st.success('Thank you :)')

    elif state.b3:
      state.like = False
      state.ph1.empty()
      state.ph2.empty()
      state.ph3.empty()
      state.submit = True
  
  if state.submit:
    state.ph1.write('Make edit up here ⤴ and')

    if state.ph2.button('Submit'):
      if different:
         
        state.ph2.empty()
        state.ph1.empty()
        state.ph3.empty()
        
        state.submit = False
        # Save Users contribution:
        if state.db is not None:
          if state.direction_choice == "English to Vietnamese":
            state.db.collection(u"envi").add({
                u'from_text': state.from_txt,
                u'model_output': state.text_to_show,
                u'user_translation': state.user_edit,
                u'time': time.time()
            })
          else:
            state.db.collection(u"vien").add({
                u'from_text': state.from_txt,
                u'model_output': state.text_to_show,
                u'user_translation': state.user_edit,
                u'time': time.time()
            })
          st.success("Your suggestion was recorded. Thank you :)")
        else:
          st.success("Google Cloud Platform access is currently disabled. Suggestions will be recorded once access is enabled.")
        
        state.user_edit = state.ph0.text_area(
                    'Translated text',
                    height=100,
                    value=state.user_edit,
                    key=2)
  

st.set_page_config(
  page_title="Better translation for Vietnamese",
  layout='wide'
)

#Sidebar 
  #resize the sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 340px;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)


file_ = open("Translation-image-02.png", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.sidebar.markdown(
    f'<div><img height=250 margin-left:0 src="data:image/png;base64,{data_url}",alt="Better Translation for Vietnamese" ></div>',
    unsafe_allow_html=True,
)

st.sidebar.subheader("""Better translation for Vietnamese""")
st.sidebar.markdown("Authors: [Chinh Ngo](http://github.com/ntkchinh/) and [Trieu Trinh](http://github.com/thtrieu/).")
st.sidebar.markdown('Read more about this work [here](https://blog.vietai.org/sat/).')

#Main body
local_css("style.css")

directions = ['English to Vietnamese',
              'Vietnamese to English']

state = SessionState.get(like=False, submit=False, first_time=True, prev_choice=None)


firebase_credential_json = "vietai-research-firebase-adminsdk.json"
if os.path.exists(firebase_credential_json):
  state.db = firestore.Client.from_service_account_json(firebase_credential_json)
else:
  state.db = None

state.direction_choice = st.selectbox('Direction', directions)


@st.cache(allow_output_mutation=True)
def init(direction_choice):
  if state.direction_choice == "English to Vietnamese":
    return (get_resource('envi_pure_tall9'), 
            'Welcome to the best ever translation project for Vietnamese !')
  else:
    return (get_resource('vien_pure_tall9'), 
            'Chào mừng bạn đến với dự án dịch tiếng Việt tốt nhất !')


state.encoder = text_encoder.SubwordTextEncoder(vocab_file)

with open(vocab_file, 'r') as f:
    state.vocab = f.read().split('\n')

(state.model, state.model_path), state.prompt = init(state.direction_choice)

if state.direction_choice != state.prev_choice and state.prev_choice != None:
  state.like = False
  state.submit = False
  state.first_time = True

state.prev_choice = state.direction_choice

write_ui()



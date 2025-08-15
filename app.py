import os, io, uuid, math, regex as re, numpy as np
from functools import lru_cache
from typing import List, Tuple
from pypdf import PdfReader
import trafilatura
from gtts import gTTS
from pydub import AudioSegment
from PIL import Image, ImageDraw, ImageFont
from langdetect import detect
import gradio as gr
import torch
from transformers import (
    pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"

@lru_cache(maxsize=1)
def get_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

@lru_cache(maxsize=8)
def get_translator(model_name):
    tok=AutoTokenizer.from_pretrained(model_name)
    mdl=AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    return tok, mdl

def chunk_text(t, target_chars=2500):
    t=re.sub(r"\s+"," ",t).strip()
    if not t: return []
    s=re.split(r"(?<=[\.\!\?])\s+", t)
    chunks=[]; cur=[]
    cur_len=0
    for sent in s:
        if cur_len+len(sent)>target_chars and cur:
            chunks.append(" ".join(cur)); cur=[sent]; cur_len=len(sent)
        else:
            cur.append(sent); cur_len+=len(sent)
    if cur: chunks.append(" ".join(cur))
    return chunks

def summarize_text(t):
    if not t.strip(): return ""
    chunks=chunk_text(t)
    sm=get_summarizer()
    outs=[]
    for c in chunks:
        r=sm(c, max_length=220, min_length=80, do_sample=False)
        outs.append(r[0]["summary_text"])
    joined=" ".join(outs)
    if len(joined)<800 and len(t)>1000:
        r=sm(t[:2800], max_length=230, min_length=90, do_sample=False)
        joined=r[0]["summary_text"]
    return joined

def easy_read(text):
    s=re.sub(r"\s+"," ",text).strip()
    if not s: return ""
    sents=re.split(r"(?<=[\.\!\?])\s+", s)
    sents=[x for x in sents if x]
    sents=sents[:8]
    bullets=["• "+x for x in sents]
    return "\n".join(bullets)

LANG_TO_MODEL={
    "Hindi":"Helsinki-NLP/opus-mt-en-hi",
    "Tamil":"Helsinki-NLP/opus-mt-en-ta",
    "Marathi":"Helsinki-NLP/opus-mt-en-mr",
    "Telugu":"Helsinki-NLP/opus-mt-en-te",
    "Bengali":"Helsinki-NLP/opus-mt-en-bn",
    "Kannada":"Helsinki-NLP/opus-mt-en-kn",
    "Gujarati":"Helsinki-NLP/opus-mt-en-gu",
    "Malayalam":"Helsinki-NLP/opus-mt-en-ml",
    "English":"__noop__"
}

def translate_en_to(text, target_lang):
    if target_lang=="English": return text
    model_name=LANG_TO_MODEL[target_lang]
    tok, mdl=get_translator(model_name)
    device=0 if torch.cuda.is_available() else -1
    inputs=tok(text, return_tensors="pt", truncation=True, max_length=768)
    if device>=0: inputs={k:v.to("cuda") for k,v in inputs.items()}; mdl=mdl.to("cuda")
    out=mdl.generate(**inputs, max_new_tokens=256, num_beams=4)
    res=tok.batch_decode(out, skip_special_tokens=True)[0]
    return res

def detect_lang(text):
    try: return detect(text)
    except: return "en"

def tts_mp3(text, lang_code):
    t=gTTS(text=text, lang=lang_code)
    fn=f"tts_{uuid.uuid4().hex}.mp3"
    t.save(fn)
    return fn

LANG_TO_TTS={
    "English":"en","Hindi":"hi","Tamil":"ta","Marathi":"mr","Telugu":"te",
    "Bengali":"bn","Kannada":"kn","Gujarati":"gu","Malayalam":"ml"
}

def make_infographic(text):
    W,H=1080,1350
    img=Image.new("RGB",(W,H),"white")
    d=ImageDraw.Draw(img)
    try:
        font_title=ImageFont.truetype("DejaVuSans-Bold.ttf",64)
        font_body=ImageFont.truetype("DejaVuSans.ttf",38)
    except:
        font_title=ImageFont.load_default()
        font_body=ImageFont.load_default()
    pad=60
    d.rounded_rectangle((pad,pad,W-pad,H-pad),radius=40,outline=(0,0,0),width=4)
    d.text((pad+40,pad+30),"Key Points",font=font_title,fill=(0,0,0))
    y=pad+140
    lines=[]
    for raw in text.split("\n"):
        t=raw.strip("• ").strip()
        if not t: continue
        lines.append("• "+t)
    lines=lines[:10]
    maxw=W-2*pad-60
    for line in lines:
        wrapped=[]
        words=line.split(" ")
        cur=""
        for w in words:
            test=(cur+" "+w).strip()
            if d.textlength(test,font=font_body)>maxw:
                if cur: wrapped.append(cur); cur=w
                else: wrapped.append(w); cur=""
            else:
                cur=test
        if cur: wrapped.append(cur)
        for wline in wrapped:
            d.text((pad+40,y),wline,font=font_body,fill=(0,0,0))
            y+=56
        y+=10
        if y>H-pad-80: break
    out=f"infographic_{uuid.uuid4().hex}.png"
    img.save(out)
    return out

def read_pdf(file_obj):
    r=PdfReader(file_obj)
    texts=[]
    for p in r.pages:
        try: texts.append(p.extract_text() or "")
        except: pass
    return "\n".join(texts)

def read_url(url):
    downloaded=trafilatura.fetch_url(url)
    if not downloaded: return ""
    extracted=trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    return extracted or ""

def unify_input(file, url, raw_text):
    if raw_text and raw_text.strip(): return raw_text
    if url and url.strip(): return read_url(url)
    if file is not None: return read_pdf(file.name)
    return ""

def process(file, url, raw_text, target_language, make_audio, make_card):
    base=unify_input(file, url, raw_text)
    if not base.strip():
        return "","", "", None, None, "No content detected."
    lang=detect_lang(base)
    if lang!="en":
        pass
    s=summarize_text(base)
    e=easy_read(s)
    t=translate_en_to(e if target_language!="English" else s, target_language)
    audio=None
    if make_audio:
        code=LANG_TO_TTS.get(target_language,"en")
        try:
            audio=tts_mp3(t if target_language!="English" else e, code)
        except:
            audio=None
    card=None
    if make_card:
        card=make_infographic(e if target_language=="English" else t)
    note="Educational tool for accessibility. Not official policy guidance."
    return s,e,t,audio,card,note

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# EcoAccess • Multilingual Accessibility for Sustainability\nUpload a PDF or paste a URL or text. Get a simple summary, easy-read bullets, translation, optional audio, and a shareable infographic.")
    with gr.Row():
        file=gr.File(label="PDF (optional)")
        url=gr.Textbox(label="Web URL (optional)")
    raw_text=gr.Textbox(label="Or paste raw text", lines=8)
    target_language=gr.Dropdown(choices=list(LANG_TO_MODEL.keys()), value="English", label="Output Language")
    with gr.Row():
        make_audio=gr.Checkbox(label="Generate Audio (TTS)", value=True)
        make_card=gr.Checkbox(label="Generate Infographic Card", value=True)
    go=gr.Button("Process")
    s_out=gr.Textbox(label="Summary")
    e_out=gr.Textbox(label="Easy-Read Bullets")
    t_out=gr.Textbox(label="Translated Output")
    audio_out=gr.Audio(label="Audio", type="filepath")
    card_out=gr.Image(label="Infographic", type="filepath")
    note=gr.Markdown()
    go.click(process, inputs=[file,url,raw_text,target_language,make_audio,make_card], outputs=[s_out,e_out,t_out,audio_out,card_out,note])

if __name__=="__main__":
    demo.launch(share=True)
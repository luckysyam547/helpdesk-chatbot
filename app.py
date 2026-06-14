import os
import streamlit as st
import torch
import torch.nn as nn
from data import PAIRS
from preprocess import load_data, pairs_to_tensors, MAX_LENGTH, SOS_token, PAD_token
from model import EncoderRNN, AttnDecoderRNN
from inference import load_model, generate_response

def auto_train():
    device = torch.device("cpu")
    pairs, vocab = load_data(PAIRS, MAX_LENGTH)
    input_tensor, target_tensor = pairs_to_tensors(pairs, vocab, MAX_LENGTH, device)
    dataset = torch.utils.data.TensorDataset(input_tensor, target_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    encoder = EncoderRNN(vocab.num_words, 128, 1, 0.1).to(device)
    decoder = AttnDecoderRNN(vocab.num_words, 128, 1, 0.1).to(device)
    enc_opt = torch.optim.Adam(encoder.parameters(), lr=0.001)
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)
    for epoch in range(50):
        for inp, tgt in loader:
            enc_opt.zero_grad()
            dec_opt.zero_grad()
            enc_out, enc_hid = encoder(inp)
            dec_out = decoder(enc_out, enc_hid, target_tensor=tgt,
                            max_length=tgt.size(1), sos_token=SOS_token,
                            teacher_forcing_ratio=0.9)
            loss = criterion(dec_out.reshape(-1, dec_out.size(-1)), tgt.reshape(-1))
            loss.backward()
            enc_opt.step()
            dec_opt.step()
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "vocab_word2index": vocab.word2index,
        "vocab_index2word": vocab.index2word,
        "vocab_num_words": vocab.num_words,
        "hidden_size": 128,
        "num_layers": 1,
        "max_length": MAX_LENGTH,
    }, "/tmp/chatbot_model.pt")

st.set_page_config(page_title="HelpDesk AI", page_icon="💬", layout="centered")
st.markdown("""
<style>
.bubble-user {
    background-color: #dcf8c6;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 5px 0;
    max-width: 70%;
    margin-left: auto;
    text-align: right;
}
.bubble-bot {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 5px 0;
    max-width: 70%;
    box-shadow: 0px 1px 1px rgba(0,0,0,0.1);
}
.header-bar {
    background-color: #075e54;
    color: white;
    padding: 12px;
    border-radius: 10px 10px 0 0;
    font-weight: bold;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-bar">💬 HelpDesk AI</div>', unsafe_allow_html=True)

@st.cache_resource
def get_model():
    if not os.path.exists("/tmp/chatbot_model.pt"):
        with st.spinner("Training model please wait 2 minutes..."):
            auto_train()
    device = torch.device("cpu")
    return load_model("/tmp/chatbot_model.pt", device)

encoder, decoder, word2index, index2word, max_length, device = get_model()

if "messages" not in st.session_state:
    st.session_state.messages = [("bot", "Hello! How can I help you today?")]

chat_html = '<div style="background-color:#e5ddd5;border-radius:10px;padding:15px;min-height:300px;">'
for sender, msg in st.session_state.messages:
    cls = "bubble-user" if sender == "user" else "bubble-bot"
    chat_html += f'<div class="{cls}">{msg}</div>'
chat_html += "</div>"
st.markdown(chat_html, unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type a message...", label_visibility="collapsed",
                                placeholder="Type a message...")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    st.session_state.messages.append(("user", user_input))
    response = generate_response(
        user_input, encoder, decoder, word2index, index2word, max_length, device)
    st.session_state.messages.append(("bot", response))
    st.rerun()

st.caption("Powered by PyTorch Seq2Seq with Luong Attention.")

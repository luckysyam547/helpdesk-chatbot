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
body { background-color: #f0f0f0; }
.header {
    background-color: #075e54;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 20px;
}
.chat-box {
    background-color: #e5ddd5;
    border-radius: 12px;
    padding: 20px;
    min-height: 400px;
    margin-bottom: 20px;
}
.user-msg {
    background-color: #dcf8c6;
    padding: 10px 15px;
    border-radius: 10px 10px 0px 10px;
    margin: 8px 0px 8px 80px;
    font-size: 15px;
    color: #000;
}
.bot-msg {
    background-color: #ffffff;
    padding: 10px 15px;
    border-radius: 10px 10px 10px 0px;
    margin: 8px 80px 8px 0px;
    font-size: 15px;
    color: #000;
    box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
}
.label-user {
    text-align: right;
    font-size: 11px;
    color: #666;
    margin-right: 5px;
}
.label-bot {
    text-align: left;
    font-size: 11px;
    color: #666;
    margin-left: 5px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header">💬 HelpDesk AI — Customer Support Chatbot</div>',
            unsafe_allow_html=True)

@st.cache_resource
def get_model():
    if not os.path.exists("/tmp/chatbot_model.pt"):
        with st.spinner("⏳ Training AI model for first time... please wait 2 minutes"):
            auto_train()
    device = torch.device("cpu")
    return load_model("/tmp/chatbot_model.pt", device)

encoder, decoder, word2index, index2word, max_length, device = get_model()

if "messages" not in st.session_state:
    st.session_state.messages = [("bot", "Hello! How can I help you today? 😊")]

chat_html = '<div class="chat-box">'
for sender, msg in st.session_state.messages:
    if sender == "user":
        chat_html += f'<div class="label-user">You</div><div class="user-msg">{msg}</div>'
    else:
        chat_html += f'<div class="label-bot">🤖 HelpDesk AI</div><div class="bot-msg">{msg}</div>'
chat_html += "</div>"
st.markdown(chat_html, unsafe_allow_html=True)

col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("", placeholder="Type your message here...",
                                label_visibility="collapsed", key="input")
with col2:
    send = st.button("Send 📤", use_container_width=True)

if send and user_input.strip():
    st.session_state.messages.append(("user", user_input))
    response = generate_response(
        user_input, encoder, decoder, word2index, index2word, max_length, device)
    st.session_state.messages.append(("bot", response))
    st.rerun()

st.markdown("---")
st.caption("🔒 Powered by PyTorch Seq2Seq Model with Luong Attention Mechanism")

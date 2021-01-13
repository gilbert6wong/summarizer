<<<<<<< HEAD
mkdir -p ~/.streamlit
=======
mkdir -p ~/.streamlit/
echo "[general]
email = \"gilbert6wong@gmail.com\"
" > ~/.streamlit/credentials.toml
>>>>>>> 5a0f5da0d9a6626c1403fae639a0dbee595d81ae
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml

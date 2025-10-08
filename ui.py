# ui.py
import streamlit as st
import requests
import json
import os

# API URL akan relatif terhadap path dasar aplikasi
API_PATH = "/api/execute_mission"

def run_ui():
    """
    Fungsi utama untuk menjalankan dan menampilkan UI Streamlit.
    """
    st.set_page_config(page_title="Architect Agent UI", layout="wide")
    st.title("🤖 Architect Agent Intelligence System")
    st.caption("Ditenagai oleh Qwen-Instruct, Cerebras, dan Modal")

    mission = st.text_area("Masukkan Misi Anda di sini:", height=100, placeholder="Contoh: Selidiki aktivitas terbaru grup ransomware 'LockBit'.")

    if st.button("🚀 Jalankan Misi"):
        if not mission:
            st.error("Misi tidak boleh kosong.")
        else:
            with st.spinner("Agen sedang bekerja... Menghubungi Arsitek untuk membuat rencana..."):
                try:
                    # Mendapatkan URL dasar dari environment Streamlit (hanya tersedia saat di-host)
                    base_url = os.environ.get("STREAMLIT_SERVER_BASE_URL", "/")
                    full_api_url = base_url.rstrip('/') + API_PATH
                    
                    headers = {"Content-Type": "application/json"}
                    payload = json.dumps({"mission": mission})
                    
                    # Menggunakan st.session_state untuk menyimpan URL agar tidak perlu request berulang
                    if 'server_url' not in st.session_state:
                         st.session_state.server_url = _get_server_url()

                    response = requests.post(st.session_state.server_url + API_PATH, data=payload, headers=headers, timeout=600)
                    response.raise_for_status()
                    
                    results = response.json()

                    st.success("Misi Selesai!")
                    st.subheader("📜 Laporan Akhir")
                    st.markdown(results.get("final_report", "Tidak ada laporan akhir yang dihasilkan."), unsafe_allow_html=True)

                    with st.expander("Lihat Log Eksekusi Rencana 📝"):
                        st.json(results.get("execution_log", []))

                except requests.exceptions.RequestException as e:
                    st.error(f"Gagal menghubungi agen backend: {e}")
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

def _get_server_url():
    """Workaround untuk mendapatkan URL publik dari dalam container Modal."""
    try:
        # Streamlit sets this env var.
        from streamlit.web.server.server_util import get_browser_server_address
        return f"http://{get_browser_server_address()}"
    except Exception:
        # Fallback untuk environment lain
        return "http://localhost:8000"

if __name__ == "__main__":
    run_ui()

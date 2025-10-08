# agent.py (Versi dengan Fallback API Key)

import os
import shelve
import json
import time
from pathlib import Path

# --- Konfigurasi Awal ---
CACHE_DIR = "/cache"
DB_DIR = Path(f"{CACHE_DIR}/db")
DOC_STORE_PATH = str(DB_DIR / "documents.db")
FAISS_INDEX_PATH = str(DB_DIR / "vectors.faiss")

class IntelligenceSystem:
    """
    Kelas ini berisi semua logika backend (otak dan alat) untuk agen intelijen.
    """
    def __init__(self):
        """
        Konstruktor kelas.
        """
        from sentence_transformers import SentenceTransformer
        import faiss
        
        print("🚀 Memuat model embedding & DB...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=CACHE_DIR)
        self.vector_dimension = 384
        try:
            if os.path.exists(FAISS_INDEX_PATH):
                self.index = faiss.read_index(FAISS_INDEX_PATH)
            else:
                DB_DIR.mkdir(parents=True, exist_ok=True)
                self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.vector_dimension))
        except Exception as e:
            print(f"⚠️ Gagal memuat index, membuat yang baru. Error: {e}")
            DB_DIR.mkdir(parents=True, exist_ok=True)
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.vector_dimension))
        print(f"✅ Sistem siap, berisi {self.index.ntotal} dokumen.")

    # --- Bagian 1: Mendefinisikan TOOLBOX ---
    
    def semantic_search(self, query: str, k: int = 3):
        """Mencari dokumen yang relevan secara semantik dari database."""
        import numpy as np
        if self.index.ntotal == 0:
            return {"status": "error", "detail": "Database kosong. Tidak ada yang bisa dicari."}
        
        query_vector = self.embedding_model.encode([query])[0]
        distances, ids = self.index.search(np.array([query_vector], dtype=np.float32), k)
        
        results = []
        with shelve.open(DOC_STORE_PATH) as doc_store:
            for i, doc_id in enumerate(ids[0]):
                if doc_id != -1 and str(doc_id) in doc_store:
                    doc_info = doc_store[str(doc_id)]
                    results.append({
                        "doc_id": int(doc_id),
                        "url": doc_info.get("url", "N/A"),
                        "score": 1 - distances[0][i]
                    })
        return {"status": "success", "results": results}

    def get_document_content(self, doc_id: int):
        """Mengambil konten teks lengkap dari sebuah dokumen berdasarkan ID-nya."""
        with shelve.open(DOC_STORE_PATH) as doc_store:
            if str(doc_id) in doc_store:
                return {"status": "success", "content": doc_store[str(doc_id)]["text"]}
            return {"status": "error", "detail": f"Dokumen dengan ID {doc_id} tidak ditemukan."}

    def summarize_with_maverick(self, text: str):
        """Membuat ringkasan singkat dari sebuah teks menggunakan model Llama Maverick."""
        prompt = "Anda adalah asisten AI yang efisien. Buat ringkasan satu paragraf yang padat dari teks berikut. Output HANYA ringkasan saja, tanpa basa-basi. Teks: " + text
        response = self._call_cerebras_with_fallback(
            model="llama-4-maverick-17b-128e-instruct",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def _call_cerebras_with_fallback(self, **kwargs):
        """
        BARU: Fungsi ini menangani logika fallback API key.
        Ia mencoba kunci utama, dan jika gagal, akan mencoba kunci kedua.
        """
        from cerebras.cloud.sdk import Cerebras
        from cerebras.cloud.sdk.errors import AuthenticationError # Import error spesifik

        api_keys = [
            os.environ.get("CEREBRAS_API_KEY"),
            os.environ.get("CEREBRAS_API_KEY_2")
        ]

        for i, key in enumerate(api_keys):
            if not key:
                print(f"API Key #{i+1} tidak ditemukan, melompati.")
                continue
            
            try:
                print(f"Mencoba API Key #{i+1}...")
                client = Cerebras(api_key=key)
                response = client.chat.completions.create(**kwargs)
                print(f"✅ Sukses dengan API Key #{i+1}.")
                return response
            except AuthenticationError as e:
                print(f"⚠️ Gagal autentikasi dengan API Key #{i+1}. Error: {e}")
                # Jika ini adalah kunci terakhir, kita akan gagal. Jika tidak, loop akan lanjut.
                if i == len(api_keys) - 1:
                    raise e # Gagal setelah mencoba semua kunci
            except Exception as e:
                # Menangani error lain yang mungkin terjadi
                print(f"❌ Terjadi error non-autentikasi dengan API Key #{i+1}: {e}")
                raise e
        
        # Jika semua kunci tidak ada atau gagal
        raise AuthenticationError("Semua API key yang disediakan gagal atau tidak ditemukan.")


    async def execute_mission(self, mission_data: dict):
        """Fungsi utama yang menjadi inti dari Agent. Menerima misi, membuat rencana, dan mengeksekusi."""
        mission = mission_data.get("mission")
        if not mission:
            return {"error": "Misi tidak boleh kosong."}

        # --- Langkah 1: Meminta Rencana dari Sang Arsitek (Qwen) ---
        architect_system_prompt = """
        Anda adalah "Architect", sebuah AI otonom yang bertugas sebagai perencana untuk agen intelijen.
        Tugas Anda adalah menerima sebuah Misi dan mengubahnya menjadi Rencana Aksi dalam format JSON.
        Rencana ini adalah daftar langkah-langkah yang harus dieksekusi.
        Anda memiliki akses ke "Toolbox" berikut:
        1. semantic_search(query: str, k: int=3): Mencari dokumen relevan. Gunakan ini untuk memulai investigasi.
        2. get_document_content(doc_id: int): Mengambil teks lengkap dari dokumen yang ditemukan.
        3. summarize_with_maverick(text: str): Meringkas teks panjang agar mudah dipahami.
        
        Output Anda WAJIB berupa JSON array-of-objects, tanpa teks penjelasan lain.
        Gunakan placeholder untuk parameter yang bergantung pada hasil langkah sebelumnya. Contoh: `{"doc_id": "RESULT_FROM_STEP_1[0].doc_id"}`
        """
        
        print(f"🏛️ Mengirim misi ke Arsitek: {mission}")
        response = self._call_cerebras_with_fallback(
            messages=[
                {"role": "system", "content": architect_system_prompt},
                {"role": "user", "content": f"Misi: {mission}"}
            ],
            model="qwen-3-235b-a22b-instruct-2507",
            stream=False, temperature=0.0
        )
        plan_str = response.choices[0].message.content
        
        try:
            if plan_str.strip().startswith("```json"):
                plan_str = plan_str.strip()[7:-3].strip()
            plan = json.loads(plan_str)
            print(f"🗺️ Rencana diterima dari Arsitek: {plan}")
        except json.JSONDecodeError:
            print(f"❌ Arsitek mengembalikan rencana dalam format tidak valid: {plan_str}")
            return {"error": "Gagal membuat rencana.", "final_report": "Maaf, saya gagal membuat rencana aksi yang valid. Coba ulangi misi dengan lebih spesifik.", "execution_log": []}
            
        # --- Langkah 2: Mengeksekusi Rencana ---
        execution_log = []
        context = {}
        
        available_tools = {
            "semantic_search": self.semantic_search,
            "get_document_content": self.get_document_content,
            "summarize_with_maverick": self.summarize_with_maverick
        }
        
        for i, step in enumerate(plan):
            tool_name = step.get("tool_name")
            parameters = step.get("parameters", {})
            
            for key, value in list(parameters.items()):
                if isinstance(value, str) and value.startswith("RESULT_FROM_STEP_"):
                    try:
                        parts = value.replace("]", "").replace("[", ".").split('.')
                        step_num = int(parts[0].split('_')[3])
                        result_key = f"step_{step_num}"
                        
                        temp_result = context[result_key]['results']
                        for part in parts[1:]:
                            if part.isdigit():
                                temp_result = temp_result[int(part)]
                            else:
                                temp_result = temp_result[part]
                        parameters[key] = temp_result
                    except Exception as e:
                        print(f"Gagal mem-parsing placeholder: {value}. Error: {e}")

            print(f"▶️ Mengeksekusi Langkah {i+1}: {tool_name} dengan parameter {parameters}")
            if tool_name in available_tools:
                tool_function = available_tools[tool_name]
                result = tool_function(**parameters)
                context[f"step_{i+1}"] = result
                execution_log.append({"step": i+1, "action": step, "result": result})
                print(f"✅ Hasil: {result}")
            else:
                execution_log.append({"step": i+1, "action": step, "result": {"status": "error", "detail": f"Alat '{tool_name}' tidak ditemukan."}})

        # --- Langkah 3: Membuat Laporan Akhir ---
        final_report_prompt = f"Anda adalah analis intelijen. Berdasarkan log eksekusi misi berikut, tulis laporan akhir yang komprehensif untuk misi awal: '{mission}'. Tulis dalam format Markdown. Log: {json.dumps(execution_log)}"
        response = self._call_cerebras_with_fallback(
            messages=[{"role": "user", "content": final_report_prompt}],
            model="qwen-3-235b-a22b-thinking-2507",
        )
        final_report = response.choices[0].message.content

        return {"mission": mission, "final_report": final_report, "execution_log": execution_log}

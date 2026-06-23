import streamlit as st
import pandas as pd
import requests
import base64
import json
from datetime import datetime
import pytz
from PIL import Image
from io import BytesIO
from openai import OpenAI
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="KU KPS Dog Watch",
    page_icon="🐕",
    layout="wide"
)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

REPO = st.secrets["GITHUB_REPO"]
BRANCH = st.secrets["GITHUB_BRANCH"]
CSV_PATH = st.secrets["CSV_PATH"]
TOKEN = st.secrets["GITHUB_TOKEN"]

BKK = pytz.timezone("Asia/Bangkok")

# กรุณาปรับพิกัดจริงของมหาวิทยาลัยภายหลัง
CAMPUS_CENTER = [14.0200, 99.9700]

ZONES = {
    "หน้าโรงอาหาร 2 / ใกล้ป้ายสถานพยาบาล": [14.0205, 99.9695],
    "แนวสวน/ต้นไม้ระหว่างทางเดิน": [14.0208, 99.9702],
    "หน้าหอพักนักศึกษา": [14.0215, 99.9688],
    "หน้าอาคารเรียนรวม": [14.0192, 99.9702],
    "สถานพยาบาล": [14.0188, 99.9688],
    "ประตูมหาวิทยาลัย": [14.0220, 99.9720],
    "ตลาด/ร้านค้า/โรงอาหาร": [14.0212, 99.9710],
}

BEHAVIORS = [
    "นอน/นั่งเฉยๆ",
    "เดินเดี่ยว",
    "เดินรวมกลุ่ม",
    "เห่า",
    "วิ่งตาม/วิ่งไล่",
    "แสดงท่าทีก้าวร้าว",
    "กัดแล้ว",
    "ไม่แน่ใจ"
]

ROUTES = [
    "ทางไปเรียน",
    "ทางไปทำงาน",
    "ใกล้โรงอาหาร",
    "ใกล้หอพัก",
    "ใกล้สถานพยาบาล",
    "ใกล้ตลาด/ร้านค้า",
    "ถนนหลักในมหาวิทยาลัย",
    "สวน/แนวต้นไม้",
    "อื่นๆ"
]

CSV_COLUMNS = [
    "timestamp_bkk",
    "reporter",
    "zone",
    "route",
    "behavior",
    "lat",
    "lon",
    "dog_count_ai",
    "confidence",
    "risk",
    "notes",
    "manual_note"
]


def now_bkk():
    return datetime.now(BKK).strftime("%Y-%m-%d %H:%M:%S")


def image_to_base64(img):
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def github_headers():
    return {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json"
    }


def github_get_csv():
    url = f"https://api.github.com/repos/{REPO}/contents/{CSV_PATH}?ref={BRANCH}"
    r = requests.get(url, headers=github_headers())

    if r.status_code == 404:
        return pd.DataFrame(columns=CSV_COLUMNS), None

    r.raise_for_status()
    data = r.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    df = pd.read_csv(BytesIO(content.encode("utf-8")))

    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df[CSV_COLUMNS], data["sha"]


def github_save_csv(df, sha):
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    encoded = base64.b64encode(csv_bytes).decode("utf-8")

    url = f"https://api.github.com/repos/{REPO}/contents/{CSV_PATH}"
    payload = {
        "message": f"update dog report {now_bkk()}",
        "content": encoded,
        "branch": BRANCH
    }

    if sha:
        payload["sha"] = sha

    r = requests.put(url, headers=github_headers(), json=payload)
    r.raise_for_status()


def ai_count_dogs(img):
    b64 = image_to_base64(img)

    prompt = """
คุณคือระบบช่วยประเมินภาพเพื่อความปลอดภัยในมหาวิทยาลัย
ให้ประเมินจำนวนสุนัขที่มองเห็นในภาพ แม้จะอยู่ไกลหรือถูกต้นไม้บังบางส่วน

ตอบเป็น JSON เท่านั้น:
{
  "dog_count": number,
  "confidence": "low/medium/high",
  "notes": "คำอธิบายภาษาไทยสั้นๆ"
}

ถ้าไม่แน่ใจ ให้ประเมินแบบระมัดระวัง ไม่เดาเกินจริง
"""

    try:
        res = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{b64}"
                        }
                    ]
                }
            ]
        )

        text = res.output_text.strip()
        return json.loads(text)

    except Exception as e:
        return {
            "dog_count": 0,
            "confidence": "low",
            "notes": f"AI วิเคราะห์ไม่สำเร็จ: {e}"
        }


def calculate_risk(dog_count, behavior):
    if behavior in ["กัดแล้ว", "วิ่งตาม/วิ่งไล่", "แสดงท่าทีก้าวร้าว"]:
        return "red"

    if dog_count >= 5:
        return "red"
    elif dog_count >= 2:
        return "yellow"
    else:
        return "green"


def risk_label(risk):
    if risk == "red":
        return "🔴 แดง: เสี่ยงสูง"
    elif risk == "yellow":
        return "🟡 เหลือง: เฝ้าระวัง"
    return "🟢 เขียว: พบประปราย"


def risk_color(risk):
    if risk == "red":
        return "red"
    elif risk == "yellow":
        return "orange"
    return "green"


st.title("🐕 KU KPS Dog Watch")
st.caption("ระบบรายงานจุดพบสุนัขจรจัดในมหาวิทยาลัย เพื่อเตือนภัยนักศึกษาและบุคลากรแบบใกล้ real-time")

st.warning(
    "กรุณาหลีกเลี่ยงการถ่ายใบหน้าบุคคลหรือข้อมูลส่วนตัว "
    "ภาพและข้อมูลใช้เพื่อความปลอดภัยในมหาวิทยาลัยเท่านั้น"
)

tab1, tab2, tab3 = st.tabs(
    ["📷 รายงานภาพสุนัข", "🗺️ แผนที่ความเสี่ยง", "📊 ข้อมูลล่าสุด"]
)

with tab1:
    st.subheader("รายงานจุดพบสุนัข")

    col1, col2 = st.columns(2)

    with col1:
        reporter = st.text_input("ชื่อผู้รายงาน / หน่วยงาน / เว้นว่างได้", "")
        zone = st.selectbox("สถานที่/โซนที่พบ", list(ZONES.keys()))
        route = st.selectbox("เส้นทางที่เกี่ยวข้อง", ROUTES)
        behavior = st.selectbox("พฤติกรรมสุนัขที่เห็น", BEHAVIORS)

    with col2:
        method = st.radio("วิธีส่งภาพ", ["เปิดกล้องถ่าย", "อัปโหลดรูป"])
        manual_note = st.text_area("บันทึกเพิ่มเติม เช่น เห็นตอน 06.45 ใกล้ป้ายโรงอาหาร", "")

    st.info(f"เวลาระบบอัตโนมัติ Bangkok time: {now_bkk()}")

    img_file = None

    if method == "เปิดกล้องถ่าย":
        img_file = st.camera_input("เปิดกล้องถ่ายภาพ")
    else:
        img_file = st.file_uploader("อัปโหลดภาพ", type=["jpg", "jpeg", "png"])

    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, caption="ภาพที่ส่งรายงาน", use_container_width=True)

        if st.button("วิเคราะห์ด้วย AI และบันทึกรายงาน", type="primary"):
            with st.spinner("AI กำลังประเมินจำนวนสุนัขในภาพ..."):
                result = ai_count_dogs(img)

            dog_count = int(result.get("dog_count", 0))
            confidence = result.get("confidence", "low")
            notes = result.get("notes", "")

            risk = calculate_risk(dog_count, behavior)
            lat, lon = ZONES[zone]

            new_row = {
                "timestamp_bkk": now_bkk(),
                "reporter": reporter,
                "zone": zone,
                "route": route,
                "behavior": behavior,
                "lat": lat,
                "lon": lon,
                "dog_count_ai": dog_count,
                "confidence": confidence,
                "risk": risk,
                "notes": notes,
                "manual_note": manual_note
            }

            try:
                df, sha = github_get_csv()
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                github_save_csv(df, sha)

                st.success("บันทึกรายงานสำเร็จ")
                st.metric("AI ประมาณจำนวนสุนัข", f"{dog_count} ตัว")
                st.write(f"ความมั่นใจ: **{confidence}**")
                st.write(f"ระดับความเสี่ยง: **{risk_label(risk)}**")
                st.write(notes)

            except Exception as e:
                st.error(f"บันทึกข้อมูลไม่สำเร็จ: {e}")

with tab2:
    st.subheader("แผนที่ความเสี่ยงล่าสุด")

    try:
        df, _ = github_get_csv()

        if df.empty:
            st.info("ยังไม่มีข้อมูลรายงาน")
        else:
            df["dog_count_ai"] = pd.to_numeric(df["dog_count_ai"], errors="coerce").fillna(0)

            latest = (
                df.groupby("zone", as_index=False)
                .agg({
                    "dog_count_ai": "sum",
                    "lat": "last",
                    "lon": "last",
                    "timestamp_bkk": "max",
                    "risk": lambda x: "red" if "red" in list(x)
                    else ("yellow" if "yellow" in list(x) else "green"),
                    "behavior": "last",
                    "route": "last"
                })
            )

            m = folium.Map(location=CAMPUS_CENTER, zoom_start=16)

            for _, row in latest.iterrows():
                color = risk_color(row["risk"])
                popup = f"""
                <b>{row['zone']}</b><br>
                ระดับ: {risk_label(row['risk'])}<br>
                จำนวนสุนัขสะสมจากรายงานล่าสุด: {int(row['dog_count_ai'])} ตัว<br>
                พฤติกรรมล่าสุด: {row['behavior']}<br>
                เส้นทาง: {row['route']}<br>
                อัปเดต: {row['timestamp_bkk']}
                """

                folium.CircleMarker(
                    location=[float(row["lat"]), float(row["lon"])],
                    radius=18,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.65,
                    popup=popup
                ).add_to(m)

            st_folium(m, width=1100, height=650)

            st.caption(
                "สีแดง = มีพฤติกรรมก้าวร้าว/วิ่งไล่/กัดแล้ว หรือพบจำนวนมาก, "
                "สีเหลือง = พบหลายตัว, สีเขียว = พบประปราย"
            )

    except Exception as e:
        st.error(f"โหลดแผนที่ไม่สำเร็จ: {e}")

with tab3:
    st.subheader("ตารางรายงานล่าสุด")

    try:
        df, _ = github_get_csv()

        if df.empty:
            st.info("ยังไม่มีข้อมูล")
        else:
            st.dataframe(
                df.sort_values("timestamp_bkk", ascending=False).head(100),
                use_container_width=True
            )

            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "ดาวน์โหลด CSV",
                data=csv,
                file_name="dog_reports.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"โหลดข้อมูลไม่สำเร็จ: {e}")
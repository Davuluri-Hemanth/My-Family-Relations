import io
import os
import re
from datetime import date, datetime
from functools import wraps

import numpy as np
from bson.binary import Binary
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient, ReturnDocument

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "").strip()
API_SECRET = os.getenv("API_SECRET", "").strip()
FACE_RECOGNITION_ENABLED = os.getenv("FACE_RECOGNITION_ENABLED", "false").lower() == "true"

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set")

client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=8000)
client.admin.command("ping")

db = client["Family_Database"]
collection = db["persons"]
counters = db["counters"]
relations_collection = db["relations"]

app = Flask(__name__)
CORS(app)


def auth_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # Authentication is deliberately simple for the migration.
        # Replace this with JWT/OAuth before public deployment.
        if API_SECRET:
            supplied = request.headers.get("X-API-KEY", "")
            if supplied != API_SECRET:
                return jsonify({"error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    return wrapper


def parse_date_string(value):
    if not value:
        return None
    value = str(value).strip()
    if value.lower() in {"n/a", "nil", "none"}:
        return None
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    return None


def safe_exact_name_query(name):
    return {"Name": {"$regex": f"^{re.escape(name.strip())}$", "$options": "i"}}


def serialize_person(doc, include_embedding=False):
    if not doc:
        return None
    result = {
        "Person_id": doc.get("Person_id"),
        "Name": doc.get("Name", ""),
        "DOB": doc.get("DOB", ""),
        "DOD": doc.get("DOD", ""),
        "Mobile": doc.get("Mobile", ""),
        "Mail_id": doc.get("Mail_id", ""),
        "Gender": doc.get("Gender", ""),
        "has_profile_pic": bool(doc.get("Profile_Pic")),
    }
    if include_embedding:
        result["Embedding"] = doc.get("Embedding")
    return result


def next_person_id():
    doc = counters.find_one_and_update(
        {"Person_id": "person_id"},
        {"$inc": {"seq_value": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return doc["seq_value"]


def calculate_age(person):
    birth = parse_date_string(person.get("DOB"))
    death = parse_date_string(person.get("DOD"))
    if death:
        if not birth:
            return "Late"
        years = death.year - birth.year - ((death.month, death.day) < (birth.month, birth.day))
        return f"Late ({years} yrs)"
    if birth:
        today = date.today()
        years = today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
        return f"{years} yrs"
    return "N/A"


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "database": "connected"})


@app.post("/api/login")
@auth_required
def login():
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    identity = str(data.get("identity", "")).strip()

    if not name or not identity:
        return jsonify({"error": "Name and mobile/email are required"}), 400

    person = collection.find_one(safe_exact_name_query(name))
    if not person:
        return jsonify({"error": "Member not found"}), 404

    mobile = str(person.get("Mobile", "")).strip()
    email = str(person.get("Mail_id", "")).strip()
    if identity != mobile and identity != email:
        return jsonify({"error": "Mobile/email does not match this member"}), 401

    result = serialize_person(person)
    result["Age"] = calculate_age(person)
    return jsonify({"person": result})


@app.get("/api/members/suggestions")
@auth_required
def suggestions():
    typed = str(request.args.get("q", "")).strip()
    if not typed:
        return jsonify({"members": []})
    query = {"Name": {"$regex": f"^{re.escape(typed)}", "$options": "i"}}
    docs = collection.find(query).limit(5)
    return jsonify({"members": [serialize_person(d) for d in docs]})


@app.post("/api/members")
@auth_required
def add_member():
    data = request.get_json(silent=True) or {}
    name = str(data.get("Name", "")).strip()
    if not name:
        return jsonify({"error": "Name is required"}), 400
    if collection.find_one(safe_exact_name_query(name)):
        return jsonify({"error": "A member with this name already exists"}), 409

    mobile = str(data.get("Mobile", "")).strip()
    mail = str(data.get("Mail_id", "")).strip()
    if not mobile or not mail:
        return jsonify({"error": "Mobile Number and Mail ID are required"}), 400

    doc = {
        "Person_id": next_person_id(),
        "Name": name,
        "DOB": str(data.get("DOB", "")).strip(),
        "DOD": str(data.get("DOD", "")).strip(),
        "Mobile": mobile,
        "Mail_id": mail,
        "Gender": str(data.get("Gender", "")).strip(),
        "Profile_Pic": None,
        "Embedding": None,
    }
    collection.insert_one(doc)
    return jsonify({"person": serialize_person(doc)}), 201


@app.get("/api/members/<path:name>")
@auth_required
def member_by_name(name):
    doc = collection.find_one(safe_exact_name_query(name))
    if not doc:
        return jsonify({"error": "Member not found"}), 404
    result = serialize_person(doc)
    result["Age"] = calculate_age(doc)
    return jsonify({"person": result})


@app.get("/api/members/<int:person_id>/connections")
@auth_required
def connections(person_id):
    rows = []
    query = {"$or": [{"person_id_1": person_id}, {"person_id_2": person_id}]}
    for conn in relations_collection.find(query):
        if conn.get("person_id_1") == person_id:
            relative_id = conn.get("person_id_2")
            relation = conn.get("relation", "Relative")
            context = f"{relation} of"
        else:
            relative_id = conn.get("person_id_1")
            relation = conn.get("relation", "Relative")
            context = f"Relative (Connected via {relation})"

        relative = collection.find_one({"Person_id": relative_id})
        if relative:
            rows.append({
                "person": serialize_person(relative),
                "relation": relation,
                "description": f"{relative.get('Name', '')} ({context})",
            })
    return jsonify({"connections": rows})


@app.post("/api/relations/find")
@auth_required
def find_relation():
    data = request.get_json(silent=True) or {}
    name1 = str(data.get("person1", "")).strip()
    name2 = str(data.get("person2", "")).strip()

    doc1 = collection.find_one(safe_exact_name_query(name1))
    doc2 = collection.find_one(safe_exact_name_query(name2))
    if not doc1 or not doc2:
        return jsonify({"error": "One or both members could not be found"}), 404

    id1, id2 = doc1["Person_id"], doc2["Person_id"]

    direct = relations_collection.find_one({"person_id_1": id1, "person_id_2": id2})
    if direct:
        return jsonify({"relation": f"{doc1['Name']} is the {direct.get('relation')} of {doc2['Name']}"})

    inverse = relations_collection.find_one({"person_id_1": id2, "person_id_2": id1})
    if inverse:
        return jsonify({"relation": f"{doc2['Name']} is the {inverse.get('relation')} of {doc1['Name']}"})

    return jsonify({"relation": "No direct relationship record found between them."})


@app.post("/api/members/<int:person_id>/photo")
@auth_required
def upload_photo(person_id):
    doc = collection.find_one({"Person_id": person_id})
    if not doc:
        return jsonify({"error": "Member not found"}), 404
    if "file" not in request.files:
        return jsonify({"error": "No image file supplied"}), 400

    raw = request.files["file"].read()
    if not raw:
        return jsonify({"error": "Empty image"}), 400

    collection.update_one(
        {"Person_id": person_id},
        {"$set": {"Profile_Pic": Binary(raw)}},
    )
    return jsonify({"ok": True})


@app.get("/api/members/<int:person_id>/photo")
@auth_required
def get_photo(person_id):
    from flask import Response
    doc = collection.find_one({"Person_id": person_id}, {"Profile_Pic": 1})
    if not doc or not doc.get("Profile_Pic"):
        return jsonify({"error": "No profile picture"}), 404
    return Response(bytes(doc["Profile_Pic"]), mimetype="image/jpeg")


@app.post("/api/face/recognize")
@auth_required
def recognize_face():
    if not FACE_RECOGNITION_ENABLED:
        return jsonify({"error": "Face recognition is disabled on this server"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No image file supplied"}), 400

    raw = request.files["file"].read()
    if not raw:
        return jsonify({"error": "Empty image"}), 400

    try:
        import cv2
        from insightface.app import FaceAnalysis
        from scipy.spatial.distance import cosine
    except Exception as exc:
        return jsonify({"error": f"Face recognition dependencies are unavailable: {exc}"}), 503

    try:
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image"}), 400

        face_app = app.config.get("FACE_APP")
        if face_app is None:
            face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            face_app.prepare(ctx_id=0, det_size=(640, 640))
            app.config["FACE_APP"] = face_app

        faces = face_app.get(image)
        if not faces:
            return jsonify({"matched": False, "message": "No face detected"})

        scanned = faces[0].normed_embedding
        best = None
        lowest = 1.0
        threshold = 0.50

        for member in collection.find({"Embedding": {"$exists": True, "$ne": None}}):
            emb = np.array(member["Embedding"])
            dist = cosine(scanned, emb)
            if dist < lowest:
                lowest = dist
                best = member

        if best is not None and lowest < threshold:
            return jsonify({
                "matched": True,
                "distance": float(lowest),
                "person": serialize_person(best),
            })
        return jsonify({"matched": False, "distance": float(lowest)})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.errorhandler(Exception)
def unhandled(error):
    app.logger.exception(error)
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5001"))
    app.run(host=host, port=port, debug=False)

import { useRef, useState } from "react";

export default function UploadPanel({ onUpload }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);

  const handleFiles = async (files) => {
    if (!files || files.length === 0) return;
    setBusy(true);
    try {
      for (const file of files) {
        await onUpload(file);
      }
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  return (
    <div
      className={`upload-panel ${dragging ? "dragging" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        handleFiles(e.dataTransfer.files);
      }}
    >
      <p className="upload-title">Importer des logs</p>
      <p className="muted">CSV · JSON · JSONL</p>
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.json,.jsonl,.ndjson"
        multiple
        hidden
        onChange={(e) => handleFiles(e.target.files)}
      />
      <button
        className="btn primary"
        disabled={busy}
        onClick={() => inputRef.current?.click()}
      >
        {busy ? "Import…" : "Choisir un fichier"}
      </button>
    </div>
  );
}

# Metode Penelitian

**Pertanyaan penelitian:** Bagaimana prapemrosesan citra dan rekonstruksi indentasi memengaruhi kualitas transkripsi Python tulisan tangan dan reliabilitas analisis sintaks?

## Variabel
- Baseline: normalisasi minimal, pretrained TrOCR.
- Preprocessing: denoise, contrast, threshold, deskew (masih direncanakan).
- Layout: tanpa rekonstruksi indentasi dibanding heuristik geometris; bobot OCR dibuat tetap.

## Metrik
- CER = (substitusi + delesi + insersi) / jumlah karakter referensi.
- Exact-line accuracy = baris yang cocok persis / seluruh baris.
- Indentation accuracy = tingkat indentasi tepat / baris teranotasi.
- AST validity agreement = kesesuaian valid/tidak valid terhadap transkripsi manual.
- Rubric MAE = rerata selisih mutlak skor kode OCR dan kode referensi; ini bukan ukuran nilai mahasiswa yang benar.

## Dataset
Kumpulkan sampel kode Python tulisan tangan dengan persetujuan penulis, termasuk kode sintaks benar dan salah. Anotasi transkripsi persis (kesalahan asli tetap ada), bounding box baris, dan tingkat indentasi. Pisahkan train, validation, dan test berdasarkan penulis, bukan acak per baris.

## Catatan
Prototype belum menerapkan kondisi eksperimen A/B terkontrol. Rubrik adalah metrik demonstrasi, bukan standar validitas sintaks parsial dan tidak boleh dipakai sebagai nilai akhir ujian. Hasil OCR harus diverifikasi sebelum dikaitkan dengan kesalahan mahasiswa.

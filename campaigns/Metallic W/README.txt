Metallic W XRD preset export
============================

Contents
--------
- Metallic_W_preset.json
  Portable preset containing the two W phase CIFs and the saved refinement
  controls.
- W_Im-3m_mp-91.cif
- W_Pm-3n_mp-11334.cif

Install
-------
The toolkit stores user presets in:

  <toolkit folder>\xrd_refinement_presets.json

If the recipient has no saved presets:

1. Back up any existing xrd_refinement_presets.json file.
2. Copy Metallic_W_preset.json into the toolkit root.
3. Rename it to xrd_refinement_presets.json.
4. Restart the toolkit and load "Metallic W" from the Presets menu.

If the recipient already has saved presets:

1. Open both JSON files in a text editor.
2. Copy the single object inside the exported file's "presets" array.
3. Append it to the existing file's "presets" array, separated by a comma.
4. Save the existing file and restart the toolkit.

The CIF files are included separately for inspection and provenance. The
portable preset also embeds their contents, so a Materials Project API key
or an existing CIF cache is not required to run this preset.

---
name: image2teletext
description: Convert an image to BBC Micro Mode 7 / Teletext format using image2teletext.py. Produces a 1000-byte .bin file and optional preview PNG. Use this when the user wants to convert, encode, or export an image as Teletext / Mode 7.
argument-hint: <image> [options]
allowed-tools: Bash(python image2teletext.py *), Bash(python -c *), Read, Glob
---

Convert an image to Teletext / Mode 7 format using `image2teletext.py`.

## Arguments

`$ARGUMENTS` — pass directly to `image2teletext.py`. At minimum an image path is required.

## Steps

1. If no arguments were given, explain usage (see below) and stop.

2. Check the input file exists. If it doesn't, say so and stop.

3. Build the command. Always include `--preview` pointing to a `.png` alongside the output `.bin` so the user can see the result:
   - If `-o` is not in the arguments, the output will be `<input>.bin` by default
   - Add `--preview <input>.preview.png` unless `--preview` is already in the arguments

4. Run the conversion from the repo root directory:
   ```
   python image2teletext.py $ARGUMENTS --preview <preview_path>
   ```

5. Report:
   - The output `.bin` path
   - The preview PNG path
   - The edit.tf and ZXNet URLs (add `--url --zxnet` if not already present and re-run, or extract from stderr if printed)
   - Any warnings or errors

## Usage examples

```
/image2teletext photo.jpg
/image2teletext photo.jpg --preset vivid
/image2teletext photo.jpg --preset art --smooth 3
/image2teletext photo.jpg -o out.bin --preview out.png --edge-weight 3 --sep
/image2teletext photo.jpg --greedy --preview quick.png
```

## Available presets

`photo`, `clean`, `smooth`, `vivid`, `graphic`, `flat`, `retro`, `art`, `dark`, `tv`, `crt`

## Key options

| Option | Effect |
|--------|--------|
| `--preset NAME` | Named settings bundle |
| `--edge-weight W` | Prioritise silhouette edges (2–5) |
| `--sep` | Separated graphics mode |
| `--smooth N` | Merge short colour runs (2–6) |
| `--direct-sample` | Bypass bilinear resize (good for hard-edged sources) |
| `--greedy` | ~4× faster, slightly lower quality |
| `--url` | Print edit.tf URL |
| `--zxnet` | Print ZXNet URL |

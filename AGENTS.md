# Repository safety rules

- The repository's `originals/` directory is permanently read-only source material.
- Never create, modify, rename, move, replace, or delete anything inside `originals/`.
- Never place restored images, work files, caches, temporary files, manifests, or any
  other generated artifacts inside `originals/`.
- Reading an explicitly selected source image from `originals/` is allowed. Every
  write target must resolve outside the protected directory, including symlinked paths.
- For `run.sh --simple`, write `<stem>_restored.<ext>` to the caller's current working
  directory, never beside the input image.
- The `originals/` subdirectory created beneath a user-selected output directory is a
  processed-output folder and is distinct from the protected repository archive.
- When work is complete, commit and push only intended project changes. Never commit
  source photographs, generated restoration artifacts, archives, or unrelated user files.

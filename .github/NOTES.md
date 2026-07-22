# Notes

For GitHub Actions, there are two main approaches:

1. Commit dist/ (most common)
   - GitHub Actions fetches code directly from your repo at runtime
   - It needs the compiled JavaScript to execute
   - Most official actions (actions/checkout, actions/setup-node, etc.) do this

2. Build on release (cleaner history)
   - Use a CI workflow that builds and creates a release
   - Push compiled code to a release branch (e.g., v1, v1.0.0)
   - Keep main clean with only source code

For option 2, you'd typically:

1. Add dist/ to .gitignore on main
2. Create a release workflow that builds and pushes to a release branch or tag

Here's a simple release workflow example:

## .github/workflows/release.yml

```yaml
name: Release

on:
  push:
    tags:
      - "v*"

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
      - run: yarn install --immutable --immutable-cache
      - run: yarn build
      - run: yarn typecheck
      - run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add -f dist/
          git commit -m "Build for ${{ github.ref_name }}"
          git push origin HEAD:refs/heads/${{ github.ref_name }}
```

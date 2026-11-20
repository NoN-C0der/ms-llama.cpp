#!/usr/bin/env bash
# Promote an MS pre-release (ms-b<N>) to a full release (ms-vX.Y[.Z]) marked "latest": the same commit, the same assets.
# Used by .github/workflows/ms-promote.yml and ms-release.yml; needs gh with GH_TOKEN (contents: write).
#
# usage: scripts/ms-release-promote.sh <version> [<pre-release tag>]     (no tag: the newest ms-b pre-release)
set -euo pipefail

VERSION="${1:?usage: ms-release-promote.sh <version> [<pre-release tag>]}"
FROM="${2:-}"
REPO="${GITHUB_REPOSITORY:-devvexus/ms-llama.cpp}"

if [[ ! "$VERSION" =~ ^ms-v[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
    echo "::error::the version must look like ms-v1.1 or ms-v1.1.2, got '$VERSION'"
    exit 1
fi
if gh release view "$VERSION" --repo "$REPO" > /dev/null 2>&1; then
    echo "::error::release $VERSION exists already"
    exit 1
fi

if [[ -z "$FROM" ]]; then
    FROM=$(gh release list --repo "$REPO" --limit 100 --json tagName,isPrerelease,createdAt \
        --jq '[.[] | select(.isPrerelease and (.tagName | startswith("ms-b")))] | sort_by(.createdAt) | last | .tagName // ""')
    if [[ -z "$FROM" ]]; then
        echo "::error::no ms-b pre-release found"
        exit 1
    fi
fi
if [[ ! "$FROM" =~ ^ms-b[0-9]+$ ]]; then
    echo "::error::the pre-release must be an ms-b<N> tag, got '$FROM'"
    exit 1
fi

# the commit of the pre-release (its tag points at the commit)
SHA=$(gh api "repos/$REPO/commits/$FROM" --jq .sha)
echo "promoting $FROM ($SHA) to $VERSION"

DIR=$(mktemp -d)
gh release download "$FROM" --repo "$REPO" --dir "$DIR"
ls -l "$DIR"

NOTES="$DIR/notes.md"
{
    echo "ms-llama.cpp $VERSION: the build of pre-release [$FROM](https://github.com/$REPO/releases/tag/$FROM), commit $SHA."
    echo
    echo "MS (Mad Shift) Gen 2 type ids (201..206). MS Gen 1 files (ms-v1.0) are migrated with"
    echo "\`python gguf-py/gguf/scripts/gguf_ms_migrate.py <gen1.gguf> <gen2.gguf>\`."
    echo
    gh release view "$FROM" --repo "$REPO" --json body --jq .body
} > "$NOTES"

gh release create "$VERSION" "$DIR"/ms-llama-* \
    --repo "$REPO" \
    --target "$SHA" \
    --title "ms-llama.cpp $VERSION" \
    --notes-file "$NOTES" \
    --latest

echo "released $VERSION: https://github.com/$REPO/releases/tag/$VERSION"

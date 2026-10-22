#!/usr/bin/env bash
# One-off: copy the local Docker database into an EMPTY Neon database.
#
# Reads NEON_DATABASE_URL from .env so the password never appears on the
# command line, in shell history, or in logs. Use Neon's direct (non-pooler)
# connection string; pg_restore needs a session the pooler doesn't give.
set -euo pipefail
cd "$(dirname "$0")/.."

url=$(grep -E '^NEON_DATABASE_URL=' .env 2>/dev/null | head -1 | cut -d= -f2- | tr -d "\"'" || true)
if [ -z "$url" ]; then
    echo "Add NEON_DATABASE_URL=postgresql://... to .env first (Neon's direct connection string)."
    exit 1
fi

pg() { docker run --rm -i postgres:17 "$@"; }

tables=$(pg psql "$url" -Atc "select count(*) from information_schema.tables where table_schema = 'public'")
if [ "$tables" != "0" ]; then
    echo "Neon database already has $tables tables — refusing to overwrite. Use an empty database."
    exit 1
fi

dump=$(mktemp -t fantasy26-dump)
trap 'rm -f "$dump"' EXIT

echo "dumping local database..."
docker compose exec -T db sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc --no-owner --no-privileges' > "$dump"
echo "  $(du -h "$dump" | cut -f1) compressed"

echo "restoring into Neon..."
pg pg_restore --no-owner --no-privileges --exit-on-error -d "$url" < "$dump"

echo "checking row counts (local vs Neon):"
q="select 'player_fixture_stats' t, count(*) from player_fixture_stats
   union all select 'player_snapshots', count(*) from player_snapshots
   union all select 'predictions', count(*) from predictions
   union all select 'alembic_version', count(*) from alembic_version
   order by 1"
paste -d'|' \
    <(docker compose exec -T db sh -c "psql -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" -At -c \"$q\"") \
    <(pg psql "$url" -At -c "$q" | cut -d'|' -f2)
echo "done. Now set DATABASE_URL in .env to the same Neon URL."

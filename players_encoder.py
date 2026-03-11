from typing import List, Optional, Dict
import polars as pl
import numpy as np

SUS_IDS = {4294967295}


class PlayerEncoder:

    DEFAULT_STAT_COLS = [
        'kills', 'deaths', 'assists',
        'gold_per_min', 'xp_per_min',
        'hero_damage', 'tower_damage',
        'last_hits', 'denies',
    ]

    DERIVED_STATS = ['kda', 'kill_death_ratio']

    def __init__(
            self,
            players_df: pl.DataFrame,
            rank_n_matches: int = 5,
            smoothing: float = 10.0,
            stat_cols: Optional[List[str]] = None,
            exclude_accounts: Optional[set] = None,
    ):
        self.players_df = players_df
        self.rank_n_matches = rank_n_matches
        self.smoothing = smoothing
        self.stat_cols = stat_cols or self.DEFAULT_STAT_COLS.copy()
        self.exclude_accounts = exclude_accounts or SUS_IDS

        self._player_stats: Optional[pl.DataFrame] = None
        self._global_stats: Optional[Dict[str, float]] = None
        self._feature_cols: Optional[List[str]] = None

        self._player_raw_winrate: Optional[pl.DataFrame] = None
        self._train_match_wins: Optional[pl.DataFrame] = None

    def fit(self, train_info: pl.DataFrame) -> 'PlayerEncoder':
        train_match_ids = train_info['match_id'].unique()

        df = (
            self.players_df
            .filter(pl.col('match_id').is_in(train_match_ids))
            .drop_nulls(subset=self.stat_cols)
        )

        if self.exclude_accounts:
            df = df.filter(~pl.col('account_id').is_in(list(self.exclude_accounts)))

        df = df.with_columns(
            pl.when(pl.col('player_slot') < 128)
            .then(pl.lit('radiant'))
            .otherwise(pl.lit('dire'))
            .alias('_team')
        )

        df = df.join(
            train_info.select(['match_id', 'radiant_win']),
            on='match_id',
            how='left',
        )

        df = df.with_columns(
            pl.when(pl.col('_team') == 'radiant')
            .then(pl.col('radiant_win').cast(pl.Float64))
            .otherwise(1.0 - pl.col('radiant_win').cast(pl.Float64))
            .alias('winrate')
        )

        df = df.with_columns([
            ((pl.col('kills') + pl.col('assists')) / (pl.col('deaths') + 1.0))
            .alias('kda'),
            (pl.col('kills').cast(pl.Float64) / (pl.col('deaths') + 1.0))
            .alias('kill_death_ratio'),
        ])

        self._player_raw_winrate = df.group_by('account_id').agg([
            pl.col('winrate').sum().alias('_loo_sum_wr'),
            pl.col('winrate').count().alias('_loo_n'),
        ])

        self._train_match_wins = df.select([
            'match_id', 'account_id', 'player_slot', 'winrate'
        ]).rename({'winrate': '_match_win'})

        all_stat_cols = self.stat_cols + self.DERIVED_STATS + ['winrate']

        agg_exprs = [pl.col('match_id').count().alias('n_matches')]
        for col in all_stat_cols:
            agg_exprs.append(pl.col(col).mean().alias(f'mean_{col}'))

        player_stats = df.group_by('account_id').agg(agg_exprs)

        feature_cols = [c for c in player_stats.columns if c.startswith('mean_')]

        global_stats = {}
        for col in feature_cols:
            vals = player_stats[col].drop_nulls()
            global_stats[col] = float(vals.mean()) if len(vals) > 0 else 0.0
        global_stats['n_matches'] = float(player_stats['n_matches'].mean())

        for col in feature_cols:
            player_stats = player_stats.with_columns(
                (
                        (pl.col('n_matches') * pl.col(col).fill_null(global_stats[col])
                         + self.smoothing * global_stats[col])
                        / (pl.col('n_matches') + self.smoothing)
                ).alias(col)
            )

        self._player_stats = player_stats
        self._global_stats = global_stats
        self._feature_cols = feature_cols

        return self

    def _apply_loo_winrate(self, players_with_stats: pl.DataFrame) -> pl.DataFrame:
        global_wr = self._global_stats['mean_winrate']

        players_with_stats = players_with_stats.join(
            self._player_raw_winrate,
            on='account_id',
            how='left',
        )

        players_with_stats = players_with_stats.join(
            self._train_match_wins,
            on=['match_id', 'account_id', 'player_slot'],
            how='left',
        )

        players_with_stats = players_with_stats.with_columns(
            pl.when(
                pl.col('_loo_sum_wr').is_not_null()
                & pl.col('_match_win').is_not_null()
            )
            .then(
                (pl.col('_loo_sum_wr') - pl.col('_match_win')
                 + self.smoothing * global_wr)
                / (pl.col('_loo_n') - 1.0 + self.smoothing)
            )
            .otherwise(pl.lit(global_wr))
            .alias('mean_winrate')
        )

        players_with_stats = players_with_stats.drop([
            '_loo_sum_wr', '_loo_n', '_match_win'
        ])

        return players_with_stats
    # ──────────────────────────────────────────────────────────

    def transform(self, df: pl.DataFrame, loo: bool = False) -> pl.DataFrame:
        match_ids = df['match_id'].unique()

        players_slim = (
            self.players_df
            .filter(pl.col('match_id').is_in(match_ids))
            .select(['match_id', 'account_id', 'player_slot'])
        )

        if self.exclude_accounts:
            players_slim = players_slim.with_columns(
                pl.when(pl.col('account_id').is_in(list(self.exclude_accounts)))
                .then(pl.lit(None))
                .otherwise(pl.col('account_id'))
                .alias('account_id')
            )

        players_slim = players_slim.with_columns(
            pl.when(pl.col('player_slot') < 128)
            .then(pl.lit('radiant'))
            .otherwise(pl.lit('dire'))
            .alias('_team')
        )

        players_with_stats = players_slim.join(
            self._player_stats, on='account_id', how='left',
        )

        for col in self._feature_cols:
            players_with_stats = players_with_stats.with_columns(
                pl.col(col).fill_null(self._global_stats[col])
            )
        players_with_stats = players_with_stats.with_columns(
            pl.col('n_matches').fill_null(0)
        )

        if loo and self._player_raw_winrate is not None:
            players_with_stats = self._apply_loo_winrate(players_with_stats)

        players_with_stats = players_with_stats.with_columns(
            (pl.col('n_matches') < self.rank_n_matches).cast(pl.Int8).alias('_is_unranked')
        )

        meta_stats = (
            players_with_stats
            .group_by(['match_id', '_team'])
            .agg([
                pl.col('_is_unranked').sum().alias('count_unranked'),
                pl.col('n_matches').mean().alias('avg_experience'),
            ])
        )

        ranked_players = players_with_stats.filter(
            pl.col('n_matches') >= self.rank_n_matches
        )

        stat_agg_exprs = []
        for col in self._feature_cols:
            stat_agg_exprs.append(pl.col(col).mean().alias(f'team_{col}'))

        team_stats_ranked = (
            ranked_players
            .group_by(['match_id', '_team'])
            .agg(stat_agg_exprs)
        )

        team_stats = meta_stats.join(
            team_stats_ranked, on=['match_id', '_team'], how='left',
        )

        for col in self._feature_cols:
            team_stats = team_stats.with_columns(
                pl.col(f'team_{col}').fill_null(self._global_stats[col])
            )

        radiant = team_stats.filter(pl.col('_team') == 'radiant').drop('_team')
        dire = team_stats.filter(pl.col('_team') == 'dire').drop('_team')

        radiant = radiant.rename({c: f'r_{c}' for c in radiant.columns if c != 'match_id'})
        dire = dire.rename({c: f'd_{c}' for c in dire.columns if c != 'match_id'})

        merged = radiant.join(dire, on='match_id', how='outer_coalesce')

        diff_exprs = []
        for col in self._feature_cols:
            r_name = f'r_team_{col}'
            d_name = f'd_team_{col}'
            if r_name in merged.columns and d_name in merged.columns:
                diff_exprs.append(
                    (pl.col(r_name) - pl.col(d_name)).alias(f'diff_{col}')
                )
        if diff_exprs:
            merged = merged.with_columns(diff_exprs)

        merged = merged.fill_null(0).fill_nan(0)

        new_cols = [c for c in merged.columns if c != 'match_id' and c not in df.columns]
        result = df.join(merged.select(['match_id'] + new_cols), on='match_id', how='left')

        for col in new_cols:
            result = result.with_columns(pl.col(col).fill_null(0).fill_nan(0))

        return result

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        self.fit(df)
        return self.transform(df, loo=True)

    def _mean_col_names(self) -> List[str]:
        all_stat = self.stat_cols + self.DERIVED_STATS + ['winrate']
        return [f'mean_{col}' for col in all_stat]

    def get_output_columns(self) -> List[str]:
        mean_cols = self._mean_col_names()
        cols = []
        for prefix in ['r_', 'd_']:
            cols.append(f'{prefix}count_unranked')
            cols.append(f'{prefix}avg_experience')
            for fc in mean_cols:
                cols.append(f'{prefix}team_{fc}')
        for fc in mean_cols:
            cols.append(f'diff_{fc}')
        return cols

    def get_feature_groups(self) -> Dict[str, List[str]]:
        all_cols = self.get_output_columns()
        return {
            'player_rank_info': [c for c in all_cols
                                 if 'unranked' in c or 'experience' in c],
            'player_winrate': [c for c in all_cols if 'winrate' in c],
            'player_combat': [c for c in all_cols
                              if any(x in c for x in ['kills', 'deaths', 'assists', 'kda', 'kill_death'])],
            'player_economy': [c for c in all_cols
                               if any(x in c for x in ['gold_per_min', 'xp_per_min', 'last_hits', 'denies'])],
            'player_damage': [c for c in all_cols
                              if any(x in c for x in ['hero_damage', 'tower_damage'])],
        }

    def get_scaling_columns(self) -> List[str]:
        return [c for c in self.get_output_columns() if 'unranked' not in c]
"""Join exploded candidate pairs with entity attributes from both sides,
producing the wide frame that ``features.compute_features`` consumes."""
import polars as pl


def build_pair_frame(scored_pairs: pl.DataFrame, s1: pl.DataFrame, others: pl.DataFrame) -> pl.DataFrame:
    s1_r = s1.select(
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("business_name").alias("s1_business_name"),
        pl.col("business_address").alias("s1_business_address"),
        pl.col("country").alias("s1_country"),
        pl.col("cleaned_name").alias("s1_cleaned_name"),
        pl.col("cleaned_addr").alias("s1_cleaned_addr"),
    )
    o_r = others.select(
        pl.col("entity_id").alias("other_entity_id"),
        pl.col("business_name").alias("o_business_name"),
        pl.col("business_address").alias("o_business_address"),
        pl.col("country").alias("o_country"),
        pl.col("cleaned_name").alias("o_cleaned_name"),
        pl.col("cleaned_addr").alias("o_cleaned_addr"),
    )
    return (
        scored_pairs.join(s1_r, on="source1_entity_id", how="inner")
        .join(o_r, on="other_entity_id", how="inner")
    )

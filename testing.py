def analyze_evaluation_results(results):
    """Provide detailed analysis of evaluation results"""
    if not results:
        print("No results to analyze")
        return

    print("DETAILED RESULTS ANALYSIS")
    print("=" * 50)

    # Overall performance
    if "aggregate-score" in results:
        score = results["aggregate-score"]
        print(f"\n🎯 FINAL SCORE: {score:.4f}")

        if score >= 0.8:
            print("   Status: Excellent performance! 🏆")
        elif score >= 0.6:
            print("   Status: Good performance 👍")
        elif score >= 0.4:
            print("   Status: Moderate performance ⚠️")
        else:
            print("   Status: Needs improvement 🔧")

    # Unlearning effectiveness
    print("\n📊 UNLEARNING ANALYSIS:")
    if "forget-set" in results:
        forget_results = results["forget-set"]
        forget_regurg = forget_results.get("overall-regurgitation-score", 0)
        forget_knowledge = forget_results.get("overall-knowledge-score", 0)

        print(f"   Forget Regurgitation: {forget_regurg:.4f} (lower is better)")
        print(f"   Forget Knowledge: {forget_knowledge:.4f} (lower is better)")

        if forget_regurg < 0.3 and forget_knowledge < 0.3:
            print(
                "   ✅ Excellent forgetting - model successfully unlearned target information"
            )
        elif forget_regurg < 0.5 and forget_knowledge < 0.5:
            print("   🟡 Good forgetting - reasonable unlearning performance")
        else:
            print("   ❌ Poor forgetting - model retains too much target information")

    # Knowledge retention
    print("\n🧠 KNOWLEDGE RETENTION:")
    if "retain-set" in results:
        retain_results = results["retain-set"]
        retain_regurg = retain_results.get("overall-regurgitation-score", 0)
        retain_knowledge = retain_results.get("overall-knowledge-score", 0)

        print(f"   Retain Regurgitation: {retain_regurg:.4f} (higher is better)")
        print(f"   Retain Knowledge: {retain_knowledge:.4f} (higher is better)")

        if retain_regurg > 0.7 and retain_knowledge > 0.7:
            print("   ✅ Excellent retention - model preserves important knowledge")
        elif retain_regurg > 0.5 and retain_knowledge > 0.5:
            print("   🟡 Good retention - acceptable knowledge preservation")
        else:
            print(
                "   ❌ Poor retention - model lost important knowledge (catastrophic forgetting)"
            )

    # General knowledge (MMLU)
    if "mmlu_average" in results:
        mmlu_score = results["mmlu_average"]
        print(f"\n📚 GENERAL KNOWLEDGE (MMLU): {mmlu_score:.4f}")

        if mmlu_score >= 0.371:  # 75% of baseline threshold
            print("   ✅ Meets general knowledge threshold")
        else:
            print("   ❌ Below general knowledge threshold - may affect final ranking")

    # MIA resistance
    if "mia_loss_acc" in results:
        mia_acc = results["mia_loss_acc"]
        print(f"\n🛡️ MIA RESISTANCE: {mia_acc:.4f}")
        print(f"   Distance from ideal (0.5): {abs(mia_acc - 0.5):.4f}")

        if abs(mia_acc - 0.5) < 0.1:
            print("   ✅ Excellent MIA resistance - balanced unlearning")
        elif abs(mia_acc - 0.5) < 0.2:
            print("   🟡 Good MIA resistance")
        else:
            if mia_acc > 0.7:
                print("   ❌ Poor MIA resistance - under-unlearning detected")
            else:
                print("   ❌ Poor MIA resistance - over-unlearning detected")

    # Task breakdown
    print("\n📋 TASK-SPECIFIC BREAKDOWN:")
    for split in ["forget-set", "retain-set"]:
        if split not in results:
            continue

        print(f"\n   {split.replace('-set', '').upper()} SET:")
        split_data = results[split]

        for key, value in split_data.items():
            if key.startswith("Task") and isinstance(value, dict):
                print(f"     {key}:")
                for metric, score in value.items():
                    print(f"       {metric}: {score:.4f}")


if __name__ == "__main__":
    import json

    # Load evaluation results from a JSON file
    with open("evaluation_results.jsonl", "r") as f:
        results = json.load(f)

    # Analyze the results
    analyze_evaluation_results(results)

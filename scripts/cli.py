#!/usr/bin/env python3
"""CLI tool for personal recommender."""

import click
import requests
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@click.group()
@click.option("--url", default="http://localhost:8080", help="API base URL")
@click.pass_context
def cli(ctx, url):
    """Personal Recommender CLI."""
    ctx.ensure_object(dict)
    ctx.obj["url"] = url


@cli.command()
@click.option("--num", "-n", default=10, help="Number of recommendations")
@click.option("--time", "-t", type=int, help="Time budget in minutes")
@click.option("--format", "-f", type=str, help="Preferred format (article/video/podcast)")
@click.pass_context
def recommend(ctx, num, time, format):
    """Get recommendations."""
    url = ctx.obj["url"]

    payload = {"num_items": num}

    context = {}
    if time:
        context["time_budget_minutes"] = time
    if format:
        context["preferred_format"] = format

    if context:
        payload["context"] = context

    try:
        response = requests.post(f"{url}/recommend", json=payload)
        response.raise_for_status()
        data = response.json()

        # Display results
        table = Table(title=f"📋 Recommendations (Session: {data['session_id'][:8]})")
        table.add_column("#", style="cyan")
        table.add_column("Title", style="green")
        table.add_column("Creator", style="yellow")
        table.add_column("Score", style="magenta")
        table.add_column("Type", style="blue")

        for i, item in enumerate(data["items"], 1):
            table.add_row(
                str(i),
                item["title"][:50],
                (item["creator"] or "Unknown")[:30],
                f"{item['score']:.3f}",
                item["media_type"],
            )

        console.print(table)

        # Show detailed info for top item
        if data["items"]:
            top = data["items"][0]
            details = Panel(
                f"[bold]{top['title']}[/bold]\n\n"
                f"Creator: {top['creator']}\n"
                f"Topics: {', '.join(top['topics'][:5])}\n"
                f"Score: {top['score']:.3f}\n"
                f"Strategy: {top['strategy']}\n\n"
                f"Summary: {top['summary'][:200]}...\n\n"
                f"[link={top['url']}]{top['url']}[/link]",
                title="🌟 Top Recommendation",
            )
            console.print("\n")
            console.print(details)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.argument("item_id")
@click.pass_context
def like(ctx, item_id):
    """Like an item."""
    url = ctx.obj["url"]

    payload = {"item_id": item_id, "event_type": "like"}

    try:
        response = requests.post(f"{url}/feedback", json=payload)
        response.raise_for_status()
        console.print(f"[green]✓ Liked item {item_id}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.argument("item_id")
@click.pass_context
def hide(ctx, item_id):
    """Hide an item."""
    url = ctx.obj["url"]

    payload = {"item_id": item_id, "event_type": "hide"}

    try:
        response = requests.post(f"{url}/feedback", json=payload)
        response.raise_for_status()
        console.print(f"[green]✓ Hidden item {item_id}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.pass_context
def stats(ctx):
    """Show user statistics."""
    url = ctx.obj["url"]

    try:
        response = requests.get(f"{url}/stats")
        response.raise_for_status()
        data = response.json()

        panel = Panel(
            f"Total Items Viewed: {data['total_items_viewed']}\n"
            f"Total Items Completed: {data['total_items_completed']}\n"
            f"Completion Rate: {data['total_items_completed'] / max(data['total_items_viewed'], 1) * 100:.1f}%\n\n"
            f"Top Topics:\n  " + "\n  ".join(data["top_topics"][:5]) + "\n\n"
            f"Top Creators:\n  " + "\n  ".join(data["top_creators"][:5]),
            title="📊 Your Statistics",
        )
        console.print(panel)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.argument("item_id")
@click.pass_context
def show(ctx, item_id):
    """Show item details."""
    url = ctx.obj["url"]

    try:
        response = requests.get(f"{url}/items/{item_id}")
        response.raise_for_status()
        item = response.json()

        console.print(Panel(
            f"[bold]{item['title']}[/bold]\n\n"
            f"Source: {item['source']}\n"
            f"Creator: {item['creator']}\n"
            f"Type: {item['media_type']}\n"
            f"Length: {item.get('length_seconds', 0) // 60} minutes\n"
            f"Topics: {', '.join(item['topics'])}\n\n"
            f"Summary:\n{item['summary']}\n\n"
            f"URL: {item['url']}",
            title=f"📄 Item: {item_id}",
        ))

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    cli()

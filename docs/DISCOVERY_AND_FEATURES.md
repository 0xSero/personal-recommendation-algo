# Discovery Mechanisms & Feature Expansion

## Current Discovery Pipeline

### How It Finds Content to Recommend

```
┌─────────────────────────────────────────────────────────┐
│           CONTENT DISCOVERY PIPELINE                     │
└─────────────────────────────────────────────────────────┘

1. INGESTION (Pull from sources)
   │
   ├─ RSS Feeds → Poll every hour → New articles
   ├─ YouTube → yt-dlp API → Channel videos
   ├─ Browser History → SQLite query → Recently visited
   ├─ Podcasts → RSS + Whisper → Transcribed episodes
   └─ Local Files → File watcher → New markdown/PDFs
   │
   ▼
2. ENRICHMENT (Understand content)
   │
   ├─ LLM Summary (zero-shot)
   ├─ Topic Extraction (from taxonomy)
   ├─ Quality Hints (novice/intermediate/advanced)
   └─ Embedding Generation (BGE-large)
   │
   ▼
3. STORAGE
   │
   ├─ Vector DB (Qdrant) → Semantic search
   └─ Relational DB (DuckDB) → Metadata/events
   │
   ▼
4. CANDIDATE GENERATION (500 items from 1M+)
   │
   ├─ Strategy 1: TASTE CENTROID (40%)
   │   └─ Query: mean(embeddings of liked items)
   │       Find: Top 200 most similar
   │
   ├─ Strategy 2: SESSION CONTEXT (30%)
   │   └─ Query: weighted_mean(last 5 items, decay=exp)
   │       Find: Top 150 continuing current thread
   │
   ├─ Strategy 3: TOPIC SLOTS (20%)
   │   └─ For each top topic:
   │       Find: Top 20 items with that topic
   │
   └─ Strategy 4: EXPLORATION (10%)
       └─ Random sample from:
           - New creators (not seen before)
           - Fresh items (< 7 days old)
           - Long-tail topics
   │
   ▼
5. RERANKING (Top 50 from 500)
   │
   ├─ Similarity Signals
   │   ├─ cosine(item, global_taste)     0.35
   │   ├─ cosine(item, session_context)  0.20
   │   └─ cross_encoder(item, query)     0.25
   │
   ├─ Quality Signals
   │   ├─ creator_completion_rate        0.08
   │   └─ source_avg_reward              0.05
   │
   ├─ Diversity Signals
   │   ├─ novelty (new creator bonus)    0.04
   │   └─ length_match (time budget)     0.03
   │
   └─ Time Signals
       ├─ recency_decay (exponential)
       └─ repetition_penalty
   │
   ▼
6. DIVERSIFICATION (MMR)
   │
   └─ Maximal Marginal Relevance
       Balance: relevance vs diversity
       λ = 0.7 (70% relevance, 30% diversity)
   │
   ▼
7. EXPLORATION BOOST
   │
   └─ Thompson Sampling (Beta distributions)
       Arms:
         - new_creators  → α=3, β=2 → 60% win rate
         - deep_dives    → α=5, β=3 → 63% win rate
         - short_sharp   → α=2, β=4 → 33% win rate
         - hot_news      → α=4, β=4 → 50% win rate

       Sample each arm → Select top 5 items from winner
   │
   ▼
8. FINAL RANKING
   │
   └─ Top 20 items returned
       With explanations: "Because you liked X and recently read Y"
```

## 🌐 What Content Can It Ingest?

### ✅ Currently Implemented

| Source | Method | Frequency | Enrichment |
|--------|--------|-----------|------------|
| **RSS Feeds** | feedparser | Hourly | LLM summary |
| **YouTube** | yt-dlp API | 6 hours | Captions + summary |
| **Browser History** | SQLite query | 12 hours | Page extract |
| **Podcasts** | RSS + download | 6 hours | Whisper transcription |
| **Local Files** | File watcher | Hourly | Direct indexing |

### 🚀 Easy to Add (< 50 lines of code each)

#### Social Media
```python
# Twitter/X
class TwitterIngester(BaseIngester):
    """Ingest from Twitter lists, bookmarks, likes."""

    async def fetch_items(self):
        # Use tweepy or twitter API
        bookmarks = twitter_client.get_bookmarks()

        for tweet in bookmarks:
            yield Item(
                id=f"twitter_{tweet.id}",
                source="twitter",
                url=f"https://twitter.com/x/status/{tweet.id}",
                title=f"Tweet by @{tweet.author}",
                text=tweet.text + self._fetch_thread(tweet),
                creator=tweet.author,
                media_type=MediaType.TWEET,
                published_at=tweet.created_at
            )
```

**Sources:**
- ✨ **Twitter/X**: Bookmarks, likes, lists, followed accounts
- ✨ **Reddit**: Saved posts, subscribed subreddits, upvoted
- ✨ **Hacker News**: Favorites, upvoted stories
- ✨ **Mastodon**: Bookmarks, favorites, followed tags
- ✨ **LinkedIn**: Saved articles, connection posts

#### Reading Apps
```python
# Pocket
class PocketIngester(BaseIngester):
    """Ingest from Pocket saved articles."""

    async def fetch_items(self):
        pocket = PocketClient(api_key)
        articles = pocket.get_all(state="unread")

        for article in articles:
            # Extract full text with trafilatura
            full_text = trafilatura.extract(article.url)

            yield Item(
                id=f"pocket_{article.id}",
                source="pocket",
                url=article.url,
                title=article.title,
                text=full_text,
                media_type=MediaType.ARTICLE,
                extras={"pocket_tags": article.tags}
            )
```

**Sources:**
- ✨ **Pocket**: Saved articles, tags
- ✨ **Instapaper**: Saved, highlights
- ✨ **Readwise**: Highlights from all sources
- ✨ **Kindle**: Highlights, notes
- ✨ **Apple Books**: Highlights, library

#### Video Platforms
```python
# Vimeo
class VimeoIngester(BaseIngester):
    """Ingest from Vimeo likes, watch later."""

    async def fetch_items(self):
        vimeo = vimeo.VimeoClient(token=token)
        likes = vimeo.get("/me/likes")

        for video in likes['data']:
            yield Item(
                id=f"vimeo_{video['uri']}",
                source="vimeo",
                url=video['link'],
                title=video['name'],
                text=video['description'],
                creator=video['user']['name'],
                media_type=MediaType.VIDEO,
                length_seconds=video['duration']
            )
```

**Sources:**
- ✨ **Vimeo**: Likes, collections, followed creators
- ✨ **TikTok**: Favorites, followed accounts (via API)
- ✨ **Twitch**: Followed channels, VODs, clips
- ✨ **Nebula**: Watch history, queue
- ✨ **CuriosityStream**: Watch history

#### Audio Platforms
```python
# Spotify Podcasts
class SpotifyIngester(BaseIngester):
    """Ingest from Spotify saved episodes."""

    async def fetch_items(self):
        sp = spotipy.Spotify(auth=token)
        episodes = sp.current_user_saved_episodes()

        for ep in episodes['items']:
            episode = ep['episode']

            # Download and transcribe
            audio_path = self._download_episode(episode['id'])
            transcript = await self.asr.transcribe(audio_path)

            yield Item(
                id=f"spotify_{episode['id']}",
                source="spotify",
                url=episode['external_urls']['spotify'],
                title=episode['name'],
                text=transcript['text'],
                creator=episode['show']['name'],
                media_type=MediaType.PODCAST,
                length_seconds=episode['duration_ms'] / 1000
            )
```

**Sources:**
- ✨ **Spotify**: Saved episodes, shows
- ✨ **Apple Podcasts**: Library, subscriptions
- ✨ **Audible**: Library, wishlist
- ✨ **SoundCloud**: Likes, reposts

#### Learning Platforms
```python
# arXiv Papers
class ArxivIngester(BaseIngester):
    """Ingest from arXiv searches, authors."""

    async def fetch_items(self):
        import arxiv

        # Search by topics user cares about
        search = arxiv.Search(
            query="cat:cs.AI OR cat:cs.LG",
            max_results=100,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        for paper in search.results():
            yield Item(
                id=f"arxiv_{paper.entry_id}",
                source="arxiv",
                url=paper.entry_id,
                title=paper.title,
                text=paper.summary,
                creator=", ".join([a.name for a in paper.authors]),
                media_type=MediaType.ARTICLE,
                topics=paper.categories,
                published_at=paper.published
            )
```

**Sources:**
- ✨ **arXiv**: Topic searches, author follows
- ✨ **Google Scholar**: Saved articles, alerts
- ✨ **Coursera**: Saved courses, completed
- ✨ **Udemy**: Library, wishlist
- ✨ **Khan Academy**: Progress, bookmarks

#### Email & Newsletters
```python
# Gmail Newsletters
class GmailIngester(BaseIngester):
    """Ingest from Gmail newsletters."""

    async def fetch_items(self):
        gmail = Gmail(credentials)

        # Get unread from newsletter senders
        messages = gmail.users().messages().list(
            userId='me',
            q='from:substack.com OR from:medium.com is:unread'
        ).execute()

        for msg in messages['messages']:
            full = gmail.users().messages().get(
                userId='me', id=msg['id'], format='full'
            ).execute()

            # Extract text
            text = self._extract_email_text(full)

            yield Item(
                id=f"email_{msg['id']}",
                source="email",
                title=self._get_subject(full),
                text=text,
                creator=self._get_sender(full),
                media_type=MediaType.ARTICLE
            )
```

**Sources:**
- ✨ **Gmail**: Newsletters, specific senders
- ✨ **Substack**: Subscriptions
- ✨ **Beehiiv**: Newsletters
- ✨ **Email digests**: Any newsletter service

#### Collaborative Tools
```python
# Notion
class NotionIngester(BaseIngester):
    """Ingest from Notion databases."""

    async def fetch_items(self):
        notion = Client(auth=token)

        # Query reading list database
        results = notion.databases.query(
            database_id=READING_LIST_DB,
            filter={"property": "Status", "select": {"equals": "To Read"}}
        )

        for page in results['results']:
            props = page['properties']

            yield Item(
                id=f"notion_{page['id']}",
                source="notion",
                url=props['URL']['url'],
                title=props['Name']['title'][0]['plain_text'],
                tags=props['Tags']['multi_select'],
                media_type=MediaType.NOTE
            )
```

**Sources:**
- ✨ **Notion**: Databases, reading lists
- ✨ **Obsidian**: Vaults, daily notes
- ✨ **Roam**: Graphs, pages
- ✨ **Slack**: Saved messages, links
- ✨ **Discord**: Bookmarks, pinned

### 🎯 Content Discovery Strategies

#### 1. **Passive Discovery** (Current)
- Poll sources on schedule
- User doesn't do anything
- System finds new content automatically

#### 2. **Active Discovery** (Can Add)
```python
# User-triggered discovery
async def discover_similar(item_id: str, num_items: int = 10):
    """Find items similar to a specific item."""
    item = db.get_item(item_id)
    item_embedding = vector_store.get_vector(item_id)

    # Search for similar
    results = vector_store.search(item_embedding, top_k=num_items)
    return results

# Topic exploration
async def explore_topic(topic: str, num_items: int = 20):
    """Deep dive into a specific topic."""
    # Get all items with this topic
    items = db.get_items_by_topic(topic)

    # Cluster by subtopics (using embeddings)
    clusters = cluster_items(items)

    # Return diverse sample
    return sample_diverse(clusters, num_items)

# Creator discovery
async def discover_creators(creator_id: str):
    """Find similar creators."""
    # Get creator's content
    items = db.get_items_by_creator(creator_id)

    # Average embedding
    creator_embedding = np.mean([
        vector_store.get_vector(item.id) for item in items
    ], axis=0)

    # Find items from different creators with similar embeddings
    similar = vector_store.search(creator_embedding, top_k=100)

    # Filter to new creators
    new_creators = set()
    for result in similar:
        item = db.get_item(result.id)
        if item.creator != creator_id:
            new_creators.add(item.creator)

    return list(new_creators)[:10]
```

#### 3. **Serendipitous Discovery** (Can Add)
```python
# Random walk through content graph
async def serendipity_walk(start_item_id: str, steps: int = 5):
    """Random walk through similar content."""
    current = start_item_id
    path = [current]

    for _ in range(steps):
        # Get 5 similar items
        similar = await discover_similar(current, num_items=5)

        # Pick random one (weighted by score)
        scores = np.array([s.score for s in similar])
        probs = scores / scores.sum()
        next_item = np.random.choice([s.id for s in similar], p=probs)

        path.append(next_item)
        current = next_item

    return path

# Cross-pollination (different topics)
async def cross_pollinate(topic1: str, topic2: str):
    """Find items that bridge two topics."""
    items1 = db.get_items_by_topic(topic1)
    items2 = db.get_items_by_topic(topic2)

    # Find items that have both topics
    bridge_items = [
        item for item in items1
        if topic2 in item.topics
    ]

    return bridge_items
```

## 🎨 Advanced Features to Add

### 1. Smart Scheduling & Context

```python
class ContextualRecommender:
    """Recommend based on time, location, device."""

    def get_context(self) -> Dict[str, Any]:
        """Determine current context."""
        hour = datetime.now().hour
        day = datetime.now().weekday()

        if 6 <= hour < 9:
            return {
                "mode": "morning_briefing",
                "time_budget": 15,  # minutes
                "preferred_formats": ["article", "news"],
                "max_length": 5 * 60  # 5 min reads
            }
        elif 12 <= hour < 13:
            return {
                "mode": "lunch_break",
                "time_budget": 30,
                "preferred_formats": ["article", "video"],
                "novelty_boost": 0.2  # More exploration
            }
        elif 18 <= hour < 22:
            return {
                "mode": "evening_deep_dive",
                "time_budget": 120,
                "preferred_formats": ["video", "podcast", "long_article"],
                "depth_boost": 0.3  # Prefer in-depth content
            }
        elif day >= 5:  # Weekend
            return {
                "mode": "weekend_exploration",
                "time_budget": 240,
                "exploration_boost": 0.4,
                "preferred_topics": ["hobby", "entertainment"]
            }
        else:
            return {"mode": "default"}

    async def recommend_contextual(self, num_items: int = 20):
        """Get recommendations with context."""
        context = self.get_context()

        # Adjust recommendation pipeline
        recommendations = await self.base_recommend(
            num_items=num_items,
            context=context
        )

        return {
            "mode": context["mode"],
            "items": recommendations,
            "explanation": self._explain_context(context)
        }
```

### 2. Multi-Modal Search

```python
class MultiModalSearch:
    """Search by image, voice, or example."""

    async def search_by_image(self, image_path: str):
        """Find content related to an image."""
        # Use CLIP or similar
        from transformers import CLIPProcessor, CLIPModel

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Get image embedding
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        image_embedding = model.get_image_features(**inputs)

        # Search in vector store
        # (Would need to store CLIP embeddings alongside text embeddings)
        results = self.clip_vector_store.search(image_embedding, top_k=20)

        return results

    async def search_by_voice(self, audio_path: str):
        """Search by voice query."""
        # Transcribe
        transcript = await self.asr.transcribe(audio_path)

        # Search
        return await self.search_by_text(transcript['text'])

    async def search_by_example(self, item_id: str, aspect: str = "topic"):
        """'More like this' search."""
        item = self.db.get_item(item_id)

        if aspect == "topic":
            # Find similar topics
            query = " ".join(item.topics)
        elif aspect == "style":
            # Find similar writing style (would need style embeddings)
            query = f"similar style to {item.creator}"
        elif aspect == "content":
            # Find similar content
            embedding = self.vector_store.get_vector(item_id)
            return self.vector_store.search(embedding, top_k=20)

        return await self.search_by_text(query)
```

### 3. Auto-Organization

```python
class AutoOrganizer:
    """Automatically organize content into collections."""

    async def create_reading_lists(self):
        """Auto-generate themed reading lists."""
        # Get all items
        items = self.db.get_all_items()

        # Cluster by topic + difficulty
        clusters = self._cluster_items(items, num_clusters=10)

        reading_lists = []
        for cluster in clusters:
            # Analyze cluster
            dominant_topics = self._get_dominant_topics(cluster)
            avg_difficulty = self._get_avg_difficulty(cluster)

            reading_lists.append({
                "title": f"{dominant_topics[0]} - {avg_difficulty}",
                "description": self._generate_description(cluster),
                "items": cluster,
                "estimated_time": sum(item.length_seconds for item in cluster) / 60
            })

        return reading_lists

    async def suggest_connections(self, item_id: str):
        """Suggest connections between items."""
        item = self.db.get_item(item_id)

        connections = {
            "prerequisites": [],  # Items to read before this
            "related": [],        # Items on same topic
            "follow_ups": [],     # Items to read after
            "contrasting": []     # Items with different perspectives
        }

        # Find prerequisites (easier items on same topic)
        if item.audience_level == "advanced":
            easier = self.db.get_items_by_topic_and_level(
                item.topics[0], "intermediate"
            )
            connections["prerequisites"] = easier[:5]

        # Find related
        embedding = self.vector_store.get_vector(item_id)
        similar = self.vector_store.search(embedding, top_k=10)
        connections["related"] = similar

        # Find follow-ups (harder items on same topic)
        if item.audience_level == "intermediate":
            harder = self.db.get_items_by_topic_and_level(
                item.topics[0], "advanced"
            )
            connections["follow_ups"] = harder[:5]

        return connections
```

### 4. Smart Notifications

```python
class SmartNotifier:
    """Intelligent notifications about new content."""

    async def check_for_alerts(self, user_id: str):
        """Check if user should be notified."""
        profile = self.db.get_user_profile(user_id)
        alerts = []

        # 1. New content from favorite creators
        for creator in profile.top_creators[:5]:
            new_items = self.db.get_recent_items_by_creator(
                creator, hours=24
            )
            if new_items:
                alerts.append({
                    "type": "favorite_creator",
                    "message": f"New from {creator}: {new_items[0].title}",
                    "items": new_items
                })

        # 2. Breaking news in favorite topics
        for topic in profile.top_topics[:3]:
            recent = self.db.get_recent_items_by_topic(topic, hours=6)
            hot_items = [
                item for item in recent
                if self._is_trending(item)  # High engagement
            ]
            if hot_items:
                alerts.append({
                    "type": "trending_topic",
                    "message": f"Trending in {topic}",
                    "items": hot_items[:3]
                })

        # 3. Forgotten saved items
        old_saves = self.db.get_saved_items_older_than(days=7)
        if old_saves:
            alerts.append({
                "type": "reminder",
                "message": f"You saved these {len(old_saves)} items to read",
                "items": old_saves[:5]
            })

        return alerts

    async def send_digest(self, user_id: str, frequency: str = "weekly"):
        """Send digest email/notification."""
        # Get top items from past week
        top_items = await self.get_top_unread(user_id, limit=10)

        # Get reading stats
        stats = self.db.get_reading_stats(user_id, days=7)

        digest = {
            "subject": f"Your weekly reading digest",
            "stats": {
                "items_read": stats['completed'],
                "time_spent": f"{stats['total_minutes']} minutes",
                "top_topic": stats['top_topic'],
                "new_creators": stats['new_creators_discovered']
            },
            "recommended": top_items,
            "insights": self._generate_insights(stats)
        }

        return digest
```

### 5. Progressive Disclosure

```python
class ProgressiveDisclosure:
    """Start with summaries, expand on demand."""

    async def get_item_layers(self, item_id: str):
        """Get item in progressive layers."""
        item = self.db.get_item(item_id)

        return {
            "layer_0_headline": {
                "title": item.title,
                "creator": item.creator,
                "time_to_read": f"{item.length_seconds // 60} min",
                "topics": item.topics[:3]
            },

            "layer_1_summary": {
                "summary": item.summary,  # 5 bullet points
                "time_to_value": item.time_to_value,
                "audience_level": item.audience_level
            },

            "layer_2_key_points": {
                "key_points": await self._extract_key_points(item),
                "related_items": await self.discover_similar(item_id, 3)
            },

            "layer_3_full_content": {
                "text": item.text,
                "highlights": await self._extract_highlights(item),
                "questions": await self._generate_questions(item)
            },

            "layer_4_deep_dive": {
                "prerequisites": await self.find_prerequisites(item),
                "follow_ups": await self.find_follow_ups(item),
                "expert_takes": await self.find_expert_commentary(item)
            }
        }

    async def _extract_key_points(self, item: Item) -> List[str]:
        """Extract key points with LLM."""
        prompt = f"""
        Extract the 3 most important key points from this content.
        Be concise and actionable.

        Title: {item.title}
        Content: {item.text[:2000]}
        """

        response = await self.llm.agenerate(prompt, max_tokens=200)
        return self._parse_bullet_points(response)
```

### 6. Quality Filtering

```python
class QualityFilter:
    """Filter content by quality metrics."""

    async def score_quality(self, item: Item) -> Dict[str, float]:
        """Multi-dimensional quality scoring."""
        scores = {}

        # 1. Creator quality (historical)
        creator_prior = self.db.get_creator_prior(item.creator)
        scores['creator'] = creator_prior.avg_completion_rate

        # 2. Content freshness
        age_days = (datetime.utcnow() - item.published_at).days
        scores['freshness'] = np.exp(-age_days / 30)  # 30-day half-life

        # 3. Depth score (use LLM)
        depth_prompt = f"""
        Rate the depth and substance of this content on a scale of 1-10.
        Consider: analysis depth, evidence quality, nuance.

        Title: {item.title}
        Summary: {item.summary}
        """
        depth_response = await self.llm.agenerate(depth_prompt, max_tokens=10)
        scores['depth'] = float(depth_response.strip()) / 10

        # 4. Clarity score
        # Use readability metrics
        from textstat import flesch_reading_ease
        scores['clarity'] = flesch_reading_ease(item.text) / 100

        # 5. Originality score
        # Check how similar to existing content
        embedding = self.vector_store.get_vector(item.id)
        similar = self.vector_store.search(embedding, top_k=10)
        avg_similarity = np.mean([s.score for s in similar[1:]])  # Exclude self
        scores['originality'] = 1.0 - avg_similarity

        # Aggregate
        scores['overall'] = np.mean(list(scores.values()))

        return scores

    async def filter_low_quality(self, items: List[Item], threshold: float = 0.6):
        """Filter out low-quality items."""
        filtered = []

        for item in items:
            scores = await self.score_quality(item)
            if scores['overall'] >= threshold:
                filtered.append((item, scores))

        # Sort by quality
        filtered.sort(key=lambda x: x[1]['overall'], reverse=True)

        return [item for item, score in filtered]
```

### 7. Voice Interface

```python
class VoiceInterface:
    """Voice-controlled recommendations."""

    async def process_voice_command(self, audio_path: str):
        """Process voice command."""
        # Transcribe
        transcript = await self.asr.transcribe(audio_path)
        command = transcript['text'].lower()

        # Parse intent
        if "what should i read" in command:
            return await self.recommend_next()

        elif "find" in command or "search" in command:
            query = command.replace("find", "").replace("search", "").strip()
            return await self.search(query)

        elif "more like" in command:
            # Assume user referring to last item
            last_item = self.session.get_last_item()
            return await self.discover_similar(last_item.id)

        elif "summarize" in command:
            # Summarize current item
            current_item = self.session.get_current_item()
            return await self.text_to_speech(current_item.summary)

        else:
            return await self.general_query(command)

    async def text_to_speech(self, text: str) -> bytes:
        """Convert text to speech."""
        # Use TTS model (e.g., Coqui TTS)
        from TTS.api import TTS

        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        audio_path = "/tmp/speech.wav"
        tts.tts_to_file(text=text, file_path=audio_path)

        with open(audio_path, 'rb') as f:
            return f.read()
```

## 🎯 Implementation Priority

### Phase 1: Quick Wins (1 week)
1. ✨ Add 5 new content sources (Twitter, Pocket, Reddit, Notion, arXiv)
2. ✨ Implement "more like this" discovery
3. ✨ Add smart notifications for favorite creators
4. ✨ Create weekly digest emails

### Phase 2: Enhancement (2 weeks)
5. ✨ Contextual recommendations (time-based)
6. ✨ Auto-organization into reading lists
7. ✨ Quality filtering
8. ✨ Progressive disclosure UI

### Phase 3: Advanced (1 month)
9. ✨ Multi-modal search (image, voice)
10. ✨ Voice interface
11. ✨ Analytics dashboard
12. ✨ Social features (share, collaborative)

All features maintain the **zero-training** principle!

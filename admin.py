#!/usr/bin/env python3
"""
Face Recognition System Administration Tool

This tool provides administrative functions for managing the face recognition database:
- List all enrolled users
- Delete enrolled users
- View user statistics
- Backup and restore database
- Clean up corrupted entries

Usage:
    python admin.py

Features:
    - Interactive menu system
    - Safe deletion with confirmation
    - Database backup before modifications
    - User statistics and enrollment info
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

# Import existing modules
from src.recognize import load_db_npz


class FaceDBAdmin:
    """Administrative interface for face recognition database"""
    
    def __init__(self, db_path: str = "data/db/face_db.npz", 
                 json_path: str = "data/db/face_db.json"):
        self.db_path = Path(db_path)
        self.json_path = Path(json_path)
        self.backup_dir = Path("data/db/backups")
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Face DB Admin initialized")
        print(f"Database: {self.db_path}")
        print(f"JSON metadata: {self.json_path}")
    
    def load_database(self) -> Dict[str, Any]:
        """Load the face database"""
        try:
            db = load_db_npz(self.db_path)
            return db
        except Exception as e:
            print(f"Error loading database: {e}")
            return {}
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load JSON metadata if available"""
        try:
            if self.json_path.exists():
                with open(self.json_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return {}
    
    def backup_database(self) -> str:
        """Create backup of current database"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"face_db_backup_{timestamp}"
        
        # Backup NPZ file
        if self.db_path.exists():
            backup_npz = self.backup_dir / f"{backup_name}.npz"
            shutil.copy2(self.db_path, backup_npz)
            print(f"Database backed up to: {backup_npz}")
        
        # Backup JSON file if exists
        if self.json_path.exists():
            backup_json = self.backup_dir / f"{backup_name}.json"
            shutil.copy2(self.json_path, backup_json)
            print(f"Metadata backed up to: {backup_json}")
        
        return backup_name
    
    def list_users(self) -> None:
        """List all enrolled users with statistics"""
        db = self.load_database()
        metadata = self.load_metadata()
        
        if not db:
            print("No users found in database.")
            return
        
        print("\n" + "="*60)
        print("ENROLLED USERS")
        print("="*60)
        
        for i, (name, embeddings) in enumerate(db.items(), 1):
            # Get embedding count
            if isinstance(embeddings, np.ndarray):
                if embeddings.ndim == 1:
                    embed_count = 1
                else:
                    embed_count = embeddings.shape[0]
            else:
                embed_count = len(embeddings) if embeddings else 0
            
            print(f"{i:2d}. {name}")
            print(f"    Embeddings: {embed_count}")
            
            # Show metadata if available
            if name in metadata:
                user_meta = metadata[name]
                if 'enrollment_date' in user_meta:
                    print(f"    Enrolled: {user_meta['enrollment_date']}")
                if 'image_count' in user_meta:
                    print(f"    Images: {user_meta['image_count']}")
                if 'notes' in user_meta:
                    print(f"    Notes: {user_meta['notes']}")
            
            print()
        
        print(f"Total users: {len(db)}")
    
    def delete_user(self, username: str) -> bool:
        """Delete a user from the database"""
        db = self.load_database()
        
        if username not in db:
            print(f"User '{username}' not found in database.")
            return False
        
        # Show user info before deletion
        print(f"\nUser to delete: {username}")
        embeddings = db[username]
        if isinstance(embeddings, np.ndarray):
            embed_count = embeddings.shape[0] if embeddings.ndim > 1 else 1
        else:
            embed_count = len(embeddings) if embeddings else 0
        print(f"Embeddings: {embed_count}")
        
        # Confirmation
        confirm = input(f"\nAre you sure you want to delete '{username}'? (yes/no): ").lower().strip()
        if confirm not in ['yes', 'y']:
            print("Deletion cancelled.")
            return False
        
        # Create backup before deletion
        backup_name = self.backup_database()
        
        try:
            # Remove from database
            del db[username]
            
            # Save updated database
            np.savez_compressed(self.db_path, **db)
            
            # Update JSON metadata if exists
            metadata = self.load_metadata()
            if username in metadata:
                del metadata[username]
                with open(self.json_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            print(f"✓ User '{username}' deleted successfully.")
            print(f"✓ Backup created: {backup_name}")
            return True
            
        except Exception as e:
            print(f"Error deleting user: {e}")
            return False
    
    def get_database_stats(self) -> None:
        """Show database statistics"""
        db = self.load_database()
        
        if not db:
            print("Database is empty.")
            return
        
        print("\n" + "="*40)
        print("DATABASE STATISTICS")
        print("="*40)
        
        total_users = len(db)
        total_embeddings = 0
        embedding_sizes = []
        
        for name, embeddings in db.items():
            if isinstance(embeddings, np.ndarray):
                if embeddings.ndim == 1:
                    count = 1
                    size = embeddings.shape[0]
                else:
                    count = embeddings.shape[0]
                    size = embeddings.shape[1] if embeddings.ndim > 1 else embeddings.shape[0]
            else:
                count = len(embeddings) if embeddings else 0
                size = len(embeddings[0]) if embeddings and len(embeddings) > 0 else 0
            
            total_embeddings += count
            if size > 0:
                embedding_sizes.append(size)
        
        print(f"Total users: {total_users}")
        print(f"Total embeddings: {total_embeddings}")
        print(f"Average embeddings per user: {total_embeddings/total_users:.1f}")
        
        if embedding_sizes:
            print(f"Embedding dimension: {embedding_sizes[0]}")
        
        # File sizes
        if self.db_path.exists():
            db_size = self.db_path.stat().st_size / 1024  # KB
            print(f"Database file size: {db_size:.1f} KB")
        
        if self.json_path.exists():
            json_size = self.json_path.stat().st_size / 1024  # KB
            print(f"Metadata file size: {json_size:.1f} KB")
    
    def search_user(self, query: str) -> None:
        """Search for users by name"""
        db = self.load_database()
        
        if not db:
            print("Database is empty.")
            return
        
        matches = [name for name in db.keys() if query.lower() in name.lower()]
        
        if not matches:
            print(f"No users found matching '{query}'")
            return
        
        print(f"\nUsers matching '{query}':")
        for name in matches:
            embeddings = db[name]
            embed_count = embeddings.shape[0] if isinstance(embeddings, np.ndarray) and embeddings.ndim > 1 else 1
            print(f"  - {name} ({embed_count} embeddings)")
    
    def cleanup_database(self) -> None:
        """Clean up corrupted or invalid entries"""
        db = self.load_database()
        
        if not db:
            print("Database is empty.")
            return
        
        print("Checking database integrity...")
        
        corrupted_users = []
        valid_users = {}
        
        for name, embeddings in db.items():
            try:
                # Check if embeddings are valid
                if isinstance(embeddings, np.ndarray):
                    if embeddings.size == 0:
                        corrupted_users.append(name)
                        continue
                    # Try to access the data
                    _ = embeddings.shape
                    valid_users[name] = embeddings
                else:
                    if not embeddings or len(embeddings) == 0:
                        corrupted_users.append(name)
                        continue
                    valid_users[name] = embeddings
                        
            except Exception as e:
                print(f"Corrupted entry found: {name} - {e}")
                corrupted_users.append(name)
        
        if not corrupted_users:
            print("✓ Database is clean - no corrupted entries found.")
            return
        
        print(f"\nFound {len(corrupted_users)} corrupted entries:")
        for name in corrupted_users:
            print(f"  - {name}")
        
        confirm = input(f"\nRemove {len(corrupted_users)} corrupted entries? (yes/no): ").lower().strip()
        if confirm not in ['yes', 'y']:
            print("Cleanup cancelled.")
            return
        
        # Create backup
        backup_name = self.backup_database()
        
        try:
            # Save cleaned database
            np.savez_compressed(self.db_path, **valid_users)
            print(f"✓ Removed {len(corrupted_users)} corrupted entries.")
            print(f"✓ Backup created: {backup_name}")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def interactive_menu(self) -> None:
        """Interactive menu system"""
        while True:
            print("\n" + "="*50)
            print("FACE RECOGNITION DATABASE ADMIN")
            print("="*50)
            print("1. List all users")
            print("2. Delete user")
            print("3. Search users")
            print("4. Database statistics")
            print("5. Cleanup database")
            print("6. Create backup")
            print("7. Exit")
            print("-" * 50)
            
            try:
                choice = input("Select option (1-7): ").strip()
                
                if choice == '1':
                    self.list_users()
                
                elif choice == '2':
                    username = input("Enter username to delete: ").strip()
                    if username:
                        self.delete_user(username)
                    else:
                        print("Invalid username.")
                
                elif choice == '3':
                    query = input("Enter search query: ").strip()
                    if query:
                        self.search_user(query)
                    else:
                        print("Invalid query.")
                
                elif choice == '4':
                    self.get_database_stats()
                
                elif choice == '5':
                    self.cleanup_database()
                
                elif choice == '6':
                    backup_name = self.backup_database()
                    print(f"Backup created: {backup_name}")
                
                elif choice == '7':
                    print("Goodbye!")
                    break
                
                else:
                    print("Invalid option. Please select 1-7.")
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main admin interface"""
    print("Face Recognition Database Administration Tool")
    print("=" * 50)
    
    try:
        admin = FaceDBAdmin()
        admin.interactive_menu()
        
    except Exception as e:
        print(f"Error initializing admin tool: {e}")


if __name__ == "__main__":
    main()